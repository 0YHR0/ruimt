"""
Based on PureJaxRL Implementation of PPO

doing homogenous first with continuous actions. Also terminate synchronously

NOTE: currently implemented using the gymnax to smax wrapper
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainStateYokai
import orbax.checkpoint
from flax.training import orbax_utils
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import wandb
import functools
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[jnp.array]
    config: Dict

    @nn.compact
    def __call__(self, hidden, obs,dones,avail_actions):
        # obs, dones, avail_actions = x
        avail_lc1, avail_lc2, avail_mc, avail_hint = avail_actions
        avail_lc1 = jnp.array(avail_lc1)
        avail_lc2 = jnp.array(avail_lc2)
        avail_mc = jnp.array(avail_mc)
        avail_hint = jnp.array(avail_hint)
        if avail_lc1.shape[0] == 1:
            avail_lc1 = jnp.squeeze(avail_lc1, axis=0)
        if avail_lc2.shape[0] == 1:
            avail_lc2 = jnp.squeeze(avail_lc2, axis=0)
        if avail_mc.shape[0] == 1:
            avail_mc = jnp.squeeze(avail_mc, axis=0)
        if avail_hint.shape[0] == 1:
            avail_hint = jnp.squeeze(avail_hint, axis=0)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean_lc1 = nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_lc1 = 1 - avail_lc1
        action_logits_lc1 = actor_mean_lc1 - (unavail_actions_lc1 * 1e10)
        pi_lc1 = distrax.Categorical(logits=action_logits_lc1)

        actor_mean_lc2 = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_lc2 = 1 - avail_lc2
        action_logits_lc2 = actor_mean_lc2 - (unavail_actions_lc2 * 1e10)
        pi_lc2 = distrax.Categorical(logits=action_logits_lc2)

        actor_mean_mc = nn.Dense(
            self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_mc = 1 - avail_mc
        action_logits_mc = actor_mean_mc - (unavail_actions_mc * 1e10)
        pi_mc = distrax.Categorical(logits=action_logits_mc)

        actor_mean_hint = nn.Dense(
            self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_hint = 1 - avail_hint
        action_logits_hint = actor_mean_hint - (unavail_actions_hint * 1e10)
        pi_hint = distrax.Categorical(logits=action_logits_hint)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, (pi_lc1, pi_lc2, pi_mc, pi_hint), jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray  # final reward
    shaped_reward: jnp.array  # intermediate reward
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # env = FlattenObservationWrapper(env) # NOTE need a batchify wrapper
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).nvec, config=config)
        rng, _rng = jax.random.split(rng)
        # init_x = (
        #     jnp.zeros(
        #         (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n)
        #     ),
        #     jnp.zeros((1, config["NUM_ENVS"])),
        #     jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n))
        # )
        init_ob = jnp.zeros(
            (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n))
        init_done = jnp.zeros((1, config["NUM_ENVS"]))
        init_avail = (jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[0])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[1])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[2])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[3])))
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_ob,init_done,init_avail)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainStateYokai.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions_0 = {key: value[0] for key, value in avail_actions.items()}
                avail_actions_1 = {key: value[1] for key, value in avail_actions.items()}
                avail_actions_2 = {key: value[2] for key, value in avail_actions.items()}
                avail_actions_3 = {key: value[3] for key, value in avail_actions.items()}
                avail_actions_0 = jax.lax.stop_gradient(
                    batchify(avail_actions_0, env.agents, config["NUM_ACTORS"])
                )
                avail_actions_1 = jax.lax.stop_gradient(
                    batchify(avail_actions_1, env.agents, config["NUM_ACTORS"])
                )
                avail_actions_2 = jax.lax.stop_gradient(
                    batchify(avail_actions_2, env.agents, config["NUM_ACTORS"])
                )
                avail_actions_3 = jax.lax.stop_gradient(
                    batchify(avail_actions_3, env.agents, config["NUM_ACTORS"])
                )
                avail_actions = (avail_actions_0, avail_actions_1, avail_actions_2, avail_actions_3)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                hstate, pi_list, value = network.apply(train_state.params, hstate, obs_batch[np.newaxis, :], last_done[np.newaxis, :], (
                avail_actions_0[np.newaxis, :], avail_actions_1[np.newaxis, :], avail_actions_2[np.newaxis, :],
                avail_actions_3[np.newaxis, :]))  # the whole ActorCritic class is a policy
                action0 = pi_list[0].sample(seed=_rng)  # choose action based on the policy
                action1 = pi_list[1].sample(seed=_rng)  # choose action based on the policy
                action2 = pi_list[2].sample(seed=_rng)  # choose action based on the policy
                action3 = pi_list[3].sample(seed=_rng)  # choose action based on the policy
                action = jnp.array([action0, action1, action2, action3])
                action = action.transpose(1, 2, 0)
                # ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :])
                # hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                # action = pi.sample(seed=_rng)
                # log_prob = pi.log_prob(action)
                log_prob0 = pi_list[0].log_prob(action0)  # the probability of choosing this action
                log_prob1 = pi_list[1].log_prob(action1)  # the probability of choosing this action
                log_prob2 = pi_list[2].log_prob(action2)  # the probability of choosing this action
                log_prob3 = pi_list[3].log_prob(action3)  # the probability of choosing this action
                log_prob = jnp.array([log_prob0, log_prob1, log_prob2, log_prob3])
                log_prob = log_prob.transpose(1, 2, 0)
                env_act0 = unbatchify(action0, env.agents, config["NUM_ENVS"],
                                      env.num_agents)  # change action to the unbatched format to input it into the env
                env_act1 = unbatchify(action1, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act2 = unbatchify(action2, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act3 = unbatchify(action3, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {}
                for key in env_act0.keys():  # Assuming all dicts have the same keys
                    env_act[key] = jnp.concatenate([env_act0[key], env_act1[key], env_act2[key], env_act3[key]], axis=1)
                # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info['returned_episode'] = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])),
                                                        info['returned_episode'])
                info['returned_episode_lengths'] = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])),
                                                                info['returned_episode_lengths'])
                info['returned_episode_returns'] = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])),
                                                                info['returned_episode_returns'])
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    batchify(info['shaped_reward'], env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions0 = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).nvec[0])
            )
            avail_actions1 = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).nvec[1])
            )
            avail_actions2 = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).nvec[2])
            )
            avail_actions3 = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).nvec[3])
            )
            _, _, last_val = network.apply(train_state.params, hstate, last_obs_batch[np.newaxis, :], last_done[np.newaxis, :],
                                        (avail_actions0, avail_actions1, avail_actions2, avail_actions3))
            last_val = last_val.squeeze()
            # ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
            # _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            # last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val,shaped_reward_coeff):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value, shaped_reward_coeff = gae_and_next_value
                    scaled_shaped_reward = shaped_reward_coeff * transition.shaped_reward
                    shaped_reward = transition.reward + scaled_shaped_reward
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        shaped_reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value, shaped_reward_coeff), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, shaped_reward_coeff),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, train_state.shaped_reward_coeff)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(params, init_hstate.transpose(),
                                                     traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
                        log_prob0 = pi[0].log_prob(traj_batch.action[:, :, 0])
                        log_prob1 = pi[1].log_prob(traj_batch.action[:, :, 1])
                        log_prob2 = pi[2].log_prob(traj_batch.action[:, :, 2])
                        log_prob3 = pi[3].log_prob(traj_batch.action[:, :, 3])

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio0 = jnp.exp(log_prob0 - traj_batch.log_prob[:, :, 0])
                        ratio1 = jnp.exp(log_prob1 - traj_batch.log_prob[:, :, 1])
                        ratio2 = jnp.exp(log_prob2 - traj_batch.log_prob[:, :, 2])
                        ratio3 = jnp.exp(log_prob3 - traj_batch.log_prob[:, :, 3])
                        ratio = ratio3 + ratio2 + ratio1 + ratio0
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy0 = pi[0].entropy().mean()
                        entropy1 = pi[1].entropy().mean()
                        entropy2 = pi[2].entropy().mean()
                        entropy3 = pi[3].entropy().mean()
                        entropy = entropy2 + entropy1 + entropy0 + entropy3
                        logratio = log_prob0 - traj_batch.log_prob[:, :, 0] +log_prob1 - traj_batch.log_prob[:, :, 1]+log_prob2 - traj_batch.log_prob[:, :, 2]+log_prob3 - traj_batch.log_prob[:, :, 3]
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(init_hstate, (-1, config["NUM_ACTORS"]))
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            init_hstate = initial_hstate[None, :].squeeze().transpose()
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            total_loss = loss_info[0]
            value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac = loss_info[1]
            metric = traj_batch.info
            rng = update_state[-1]
            def callback(metric, train_state, traj_batch, advantages, targets, total_loss, value_loss, loss_actor,
                         entropy):
                wandb.log(
                    {
                         "total_rewards":  train_state.shaped_reward_coeff*traj_batch.shaped_reward.sum(axis=0).mean(axis=-1) + traj_batch.reward.sum(axis=0).mean(axis=-1), #  [jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                        "total_rewards_per_game_estimate":  (train_state.shaped_reward_coeff*traj_batch.shaped_reward.sum(axis=0).mean(axis=-1) + traj_batch.reward.sum(axis=0).mean(axis=-1)) / (traj_batch.reward.shape[0]/14),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        "shaped_coefficient": train_state.shaped_reward_coeff,
                        "scaled_shaped_reward": train_state.shaped_reward_coeff*traj_batch.shaped_reward.sum(axis=0).mean(),
                        "final_reward": traj_batch.reward.sum(axis=0).mean(axis=-1), #[jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                        "final_reward_per_game_estimate": traj_batch.reward.sum(axis=0).mean(axis=-1) / (traj_batch.reward.shape[0]//14), #[jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                        "advantages":advantages.sum(axis=0).mean(),
                        "targets":targets.sum(axis=0).mean(),
                        "total_loss":total_loss.sum(axis=0).mean(),
                        "value_loss":value_loss.sum(axis=0).mean(),
                        "loss_actor":loss_actor.sum(axis=0).mean(),
                        "entropy": entropy.sum(axis=0).mean()

                    }
                )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric, train_state, traj_batch, advantages, targets,
                                         total_loss, value_loss, loss_actor, entropy)
            update_steps = update_steps + 1
            new_shaped_reward_coeff_value = 1.0
            #new_shaped_reward_coeff_value = jnp.maximum(0.0, 1 - (update_steps * config["NUM_ENVS"] * config["NUM_STEPS"] / 12800000))# config["TOTAL_TIMESTEPS"]
            new_shaped_reward_coeff = jnp.full(
                train_state.shaped_reward_coeff.shape, fill_value=new_shaped_reward_coeff_value)
            jax.debug.print("Shaped reward coeff: {a}, real_dsteps: {b}, shaped_reward_steps: {c}",
                            a=new_shaped_reward_coeff, b=update_steps * config["NUM_ENVS"] * config["NUM_STEPS"],
                            c=12800000)
            # runner_state[1] is the training state object where the shaped reward coefficient is stored
            train_state = train_state.set_new_shaped_reward_coeff(
                new_coeff=new_shaped_reward_coeff)
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), init_hstate, _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_yokai")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    # checkpoint model for greedy eval
    '''train_state = out['runner_state'][0]
    save_args = orbax_utils.save_args_from_target(train_state)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_loc = 'tmp/orbax/checkpoint_rnn'
    orbax_checkpointer.save(save_loc, train_state, save_args=save_args)'''

if __name__ == "__main__":
    main()
