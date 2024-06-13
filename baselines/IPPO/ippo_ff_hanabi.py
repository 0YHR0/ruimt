"""
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from jaxmarl.wrappers.baselines import LogWrapper
import jaxmarl
import wandb
import functools
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf


class ActorCritic(nn.Module):
    action_dim: Sequence[jnp.array]
    config: Dict

    @nn.compact
    def __call__(self, obs,dones,avail_actions): # obs,dones,avail_actions
        # obs = x[0]
        # dones = x[1]
        # avail_actions = x[2:]

        avail_lc1, avail_lc2 , avail_mc, avail_hint = avail_actions
        avail_lc1 = jnp.array(avail_lc1)
        avail_lc2 = jnp.array(avail_lc2)
        avail_mc = jnp.array(avail_mc)
        avail_hint = jnp.array(avail_hint)
        if avail_lc1.shape[0]==1:
            avail_lc1= jnp.squeeze(avail_lc1, axis=0)
        if avail_lc2.shape[0]==1:
            avail_lc2= jnp.squeeze(avail_lc2, axis=0)
        if avail_mc.shape[0]==1:
            avail_mc= jnp.squeeze(avail_mc, axis=0)
        if avail_hint.shape[0]==1:
            avail_hint= jnp.squeeze(avail_hint, axis=0)
        embedding = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)
        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
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
        action_logits_mc= actor_mean_mc - (unavail_actions_mc * 1e10)
        pi_mc = distrax.Categorical(logits=action_logits_mc)

        actor_mean_hint = nn.Dense(
             self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_hint = 1 - avail_hint
        action_logits_hint = actor_mean_hint - (unavail_actions_hint * 1e10)
        pi_hint= distrax.Categorical(logits=action_logits_hint)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return (pi_lc1, pi_lc2, pi_mc, pi_hint), jnp.squeeze(critic, axis=-1)
        #return pi_list,jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
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

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).nvec, config=config)
        rng, _rng = jax.random.split(rng)
        init_ob = jnp.zeros(
            (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n))
        init_done = jnp.zeros((1, config["NUM_ENVS"]))
        init_avail = (jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[0])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[1])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[2])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[3])))
        # init_x = (
        #     jnp.zeros(
        #         (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n)
        #     ),
        #     jnp.zeros((1, config["NUM_ENVS"])),
        #
        #     jnp.zeros((4, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[0])),
        #
        # )
        #network_params = network.init(_rng, init_x)
        network_params = network.init(_rng, init_ob,init_done,init_avail)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV copy this code directly
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, rng = runner_state
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
                # avail_actions = jnp.array([avail_actions_0,avail_actions_1,avail_actions_2,avail_actions_3])
                # avail_actions = avail_actions.transpose(1,0,2)
                avail_actions = (avail_actions_0,avail_actions_1,avail_actions_2,avail_actions_3)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                #ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], (avail_actions_0[np.newaxis, :],avail_actions_1[np.newaxis, :],avail_actions_2[np.newaxis, :],avail_actions_3[np.newaxis, :]))
                pi_list, value = network.apply(train_state.params, obs_batch[np.newaxis, :],last_done[np.newaxis, :], (avail_actions_0[np.newaxis, :],avail_actions_1[np.newaxis, :],avail_actions_2[np.newaxis, :],avail_actions_3[np.newaxis, :]))#the whole ActorCritic class is a policy
                action0 = pi_list[0].sample(seed=_rng)# choose action based on the policy
                action1 = pi_list[1].sample(seed=_rng)  # choose action based on the policy
                action2 = pi_list[2].sample(seed=_rng)  # choose action based on the policy
                action3 = pi_list[3].sample(seed=_rng)  # choose action based on the policy
                action = jnp.array([action0,action1,action2,action3])
                action = action.transpose(1,2,0)
                log_prob0 = pi_list[0].log_prob(action0)# the probability of choosing this action
                log_prob1 = pi_list[1].log_prob(action1)  # the probability of choosing this action
                log_prob2 = pi_list[2].log_prob(action2)  # the probability of choosing this action
                log_prob3 = pi_list[3].log_prob(action3)  # the probability of choosing this action
                log_prob = jnp.array([log_prob0,log_prob1,log_prob2,log_prob3])
                log_prob = log_prob.transpose(1,2,0)
                env_act0 = unbatchify(action0, env.agents, config["NUM_ENVS"], env.num_agents) # change action to the unbatched format to input it into the env
                env_act1 = unbatchify(action1, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act2 = unbatchify(action2, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act3 = unbatchify(action3, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {}
                for key in env_act0.keys():  # Assuming all dicts have the same keys
                    env_act[key] = jnp.concatenate([env_act0[key], env_act1[key], env_act2[key], env_act3[key]], axis=1)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                ###17.04
                # for i in reward.keys():
                #     reward[i] = reward[i] * done[i] + reward[i]* (1-done[i]) * new_shaped_reward_coeff_value
                # info['returned_episode_returns'] = info['returned_episode_returns'].at[:,0].set(info['returned_episode_returns'][:, 0] * done['agent_0'] + info['returned_episode_returns'][:, 0] * (1 - done['agent_0'])* new_shaped_reward_coeff_value)
                # info['returned_episode_returns'] = info['returned_episode_returns'].at[:, 1].set(info['returned_episode_returns'][:, 1] * done['agent_1'] + \
                #                                          info['returned_episode_returns'][:, 1] * (
                #                                                      1 - done['agent_1']) * new_shaped_reward_coeff_value)

                #reward = reward * new_shaped_reward_coeff_value
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions
                )

                runner_state = (train_state, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            # assume all actions are legal
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
            #ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions0,avail_actions1,avail_actions2,avail_actions3)
            _, last_val = network.apply(train_state.params, last_obs_batch[np.newaxis, :], last_done[np.newaxis, :],(avail_actions0,avail_actions1,avail_actions2,avail_actions3))
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params,
                                                  traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
                        log_prob0 = pi[0].log_prob(traj_batch.action[:,:,0])
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
                        # Todo may not calculate average
                        ratio0 = jnp.exp(log_prob0 - traj_batch.log_prob[:, :, 0])
                        ratio1 = jnp.exp(log_prob1 - traj_batch.log_prob[:, :, 1])
                        ratio2 = jnp.exp(log_prob2 - traj_batch.log_prob[:, :, 2])
                        ratio3 = jnp.exp(log_prob3 - traj_batch.log_prob[:, :, 3])
                        ratio = (ratio3+ratio2+ratio1+ratio0)/4
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
                        entropy = (entropy2+entropy1+entropy0+entropy3)/4
                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
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
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            is_update = update_steps == 1

            def _true(metric):
                jax.debug.breakpoint()

            def _false(metric):
                pass

            jax.lax.cond(is_update, _true, _false,metric)
            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                    }
                )
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            jax.debug.print('-------------\nmetrics mean return = {a}', a=metric["returned_episode_returns"][-1, :].mean())
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_yokai")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(50)
    train_jit = jax.jit(make_train(config), device=jax.devices()[0])
    out = train_jit(rng)


if __name__ == "__main__":
    main()
    '''results = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    jnp.save('hanabi_results', results)
    plt.plot(results)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')'''