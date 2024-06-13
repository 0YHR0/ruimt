"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import chex

import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from flax.training.train_state import TrainStateYokai
import wandb
import functools
import matplotlib.pyplot as plt


class YokaiWorldStateWrapper(JaxMARLWrapper):

    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs, state):
        """
        For each agent: [agent obs, own hand]
        """
        all_obs = jnp.array([obs[agent] for agent in self._env.agents])
        world = state.yokai_card_world.flatten()
        world = jnp.tile(world,(self._env.num_agents,1))
        return jnp.concatenate((all_obs, world), axis=1)

    def world_state_size(self):
        result = self._env.observation_space(self._env.agents[0]).n
        result = result.item() + 1024
        return result  # NOTE hardcoded hand size


class ActorFF(nn.Module):
    action_dim: Sequence[jnp.array]
    activation: str = "relu"
    layer_dim: int = 512

    @nn.compact
    def __call__(self, obs,avail_actions):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        avail_lc1, avail_lc2 , avail_mc, avail_hint = avail_actions
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
        actor_mean = nn.Dense(
            self.layer_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.layer_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)

        action_logits_lc1 = nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_lc1 = 1 - avail_lc1
        action_logits_lc1 = action_logits_lc1 - (unavail_actions_lc1 * 1e10)
        pi_lc1 = distrax.Categorical(logits=action_logits_lc1)

        action_logits_lc2 = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_lc2 = 1 - avail_lc2
        action_logits_lc2 = action_logits_lc2 - (unavail_actions_lc2 * 1e10)
        pi_lc2 = distrax.Categorical(logits=action_logits_lc2)

        action_logits_mc = nn.Dense(
            self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_mc = 1 - avail_mc
        action_logits_mc = action_logits_mc - (unavail_actions_mc * 1e10)
        pi_mc = distrax.Categorical(logits=action_logits_mc)

        action_logits_hint = nn.Dense(
            self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions_hint = 1 - avail_hint
        action_logits_hint = action_logits_hint - (unavail_actions_hint * 1e10)
        pi_hint = distrax.Categorical(logits=action_logits_hint)

        # action_logits = nn.Dense(
        #     self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        # )(actor_mean)
        # unavail_actions = 1 - avail_actions
        # action_logits = action_logits - (unavail_actions * 1e10)
        # pi = distrax.Categorical(logits=action_logits)

        return (pi_lc1, pi_lc2, pi_mc, pi_hint)


class CriticFF(nn.Module):
    activation: str = "relu"
    layer_dim: int = 512

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            self.layer_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.layer_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(critic)

        return jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    shaped_reward: jnp.array  # intermediate reward
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    # print('batchify x', x.shape)
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
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = YokaiWorldStateWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorFF(
            env.action_space(env.agents[0]).nvec,
            activation=config["ACTIVATION"],
            layer_dim=config["LAYER_WIDTH"],
        )
        critic_network = CriticFF(
            activation=config["ACTIVATION"],
            layer_dim=config["LAYER_WIDTH"],
        )
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        init_ob = jnp.zeros(
            (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n))
        init_avail = (jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[0])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[1])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[2])),
                      jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).nvec[3])))
        # ac_init_x = (
        #     jnp.zeros((env.observation_space(env.agents[0]).n,)),
        #     jnp.zeros((env.action_space(env.agents[0]).n,)),
        # )
        actor_network_params = actor_network.init(_rng_actor, init_ob ,init_avail)

        cr_init_x = jnp.zeros((env.world_state_size(),))

        critic_network_params = critic_network.init(_rng_critic, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainStateYokai.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainStateYokai.create(
            apply_fn=actor_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, rng = runner_state

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
                # avail_actions = jax.lax.stop_gradient(
                #     batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                # )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # ac_in = (
                #     obs_batch[np.newaxis, :],
                #     avail_actions[None, :],
                # )
                pi_list = actor_network.apply(train_states[0].params,obs_batch[np.newaxis, :],(avail_actions_0[np.newaxis, :],avail_actions_1[np.newaxis, :],avail_actions_2[np.newaxis, :],avail_actions_3[np.newaxis, :]))
                action0 = pi_list[0].sample(seed=_rng)  # choose action based on the policy
                action1 = pi_list[1].sample(seed=_rng)  # choose action based on the policy
                action2 = pi_list[2].sample(seed=_rng)  # choose action based on the policy
                action3 = pi_list[3].sample(seed=_rng)  # choose action based on the policy
                action = jnp.array([action0, action1, action2, action3])
                action = action.transpose(1, 2, 0)
                # action = pi.sample(seed=_rng)
                log_prob0 = pi_list[0].log_prob(action0)  # the probability of choosing this action
                log_prob1 = pi_list[1].log_prob(action1)  # the probability of choosing this action
                log_prob2 = pi_list[2].log_prob(action2)  # the probability of choosing this action
                log_prob3 = pi_list[3].log_prob(action3)  # the probability of choosing this action
                log_prob = jnp.array([log_prob0, log_prob1, log_prob2, log_prob3])
                log_prob = log_prob.transpose(1, 2, 0)
                # log_prob = pi.log_prob(action)
                env_act0 = unbatchify(action0, env.agents, config["NUM_ENVS"],env.num_agents)
                env_act1 = unbatchify(action1, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act2 = unbatchify(action2, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act3 = unbatchify(action3, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {}
                for key in env_act0.keys():  # Assuming all dicts have the same keys
                    env_act[key] = jnp.concatenate([env_act0[key], env_act1[key], env_act2[key], env_act3[key]], axis=1)

                # env_act = unbatchify(
                #     action, env.agents, config["NUM_ENVS"], env.num_agents
                # )

                # VALUE
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))
                value = critic_network.apply(train_states[1].params, world_state)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info['returned_episode'] = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])),
                                                        info['returned_episode'])
                info['returned_episode_lengths'] = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])),
                                                                info['returned_episode_lengths'])
                info['returned_episode_returns'] = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])),
                                                                info['returned_episode_returns'])
                # info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    batchify(info['shaped_reward'], env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions
                )
                # transition = Transition(
                #     done_batch,
                #     action.squeeze(),
                #     value.squeeze(),
                #     batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                #     log_prob.squeeze(),
                #     obs_batch,
                #     world_state,
                #     info,
                #     avail_actions,
                # )
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, rng = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))
            last_val = critic_network.apply(train_states[1].params, last_world_state)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val,shaped_reward_coeff):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value,shaped_reward_coeff = gae_and_next_value
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

            advantages, targets = _calculate_gae(traj_batch, last_val, train_states[1].shaped_reward_coeff)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        # RERUN NETWORK
                        pi = actor_network.apply(
                            actor_params,
                            traj_batch.obs, traj_batch.avail_actions,
                        )
                        log_prob0 = pi[0].log_prob(traj_batch.action[:, :, 0])
                        log_prob1 = pi[1].log_prob(traj_batch.action[:, :, 1])
                        log_prob2 = pi[2].log_prob(traj_batch.action[:, :, 2])
                        log_prob3 = pi[3].log_prob(traj_batch.action[:, :, 3])
                        # log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        ratio0 = jnp.exp(log_prob0 - traj_batch.log_prob[:, :, 0])
                        ratio1 = jnp.exp(log_prob1 - traj_batch.log_prob[:, :, 1])
                        ratio2 = jnp.exp(log_prob2 - traj_batch.log_prob[:, :, 2])
                        ratio3 = jnp.exp(log_prob3 - traj_batch.log_prob[:, :, 3])
                        ratio = ratio3 + ratio2 + ratio1 + ratio0
                        # ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
                        loss_actor = loss_actor.mean(where=(1 - traj_batch.done))
                        entropy0 = pi[0].entropy().mean(where=(1 - traj_batch.done))
                        entropy1 = pi[1].entropy().mean(where=(1 - traj_batch.done))
                        entropy2 = pi[2].entropy().mean(where=(1 - traj_batch.done))
                        entropy3 = pi[3].entropy().mean(where=(1 - traj_batch.done))
                        entropy = entropy2 + entropy1 + entropy0 + entropy3
                        # entropy = pi.entropy().mean(where=(1 - traj_batch.done))
                        actor_loss = (
                                loss_actor
                                - config["ENT_COEF"] * entropy
                        )
                        return actor_loss, (loss_actor, entropy)

                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        # RERUN NETWORK
                        value = critic_network.apply(critic_params, traj_batch.world_state)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=(1 - traj_batch.done))
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, traj_batch, targets
                    )

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "critic_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                batch = (
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
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

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            total_loss = loss_info['total_loss']
            actor_loss= loss_info['actor_loss']
            critic_loss= loss_info['critic_loss']
            entropy = loss_info['entropy']

            train_states = update_state[0]
            actor_train_state, critic_train_state = train_states
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric,critic_train_state,traj_batch,advantages, targets,total_loss,actor_loss,critic_loss,entropy):
                wandb.log(
                    {
                        "total_rewards": critic_train_state.shaped_reward_coeff * traj_batch.shaped_reward.sum(axis=0).mean(
                            axis=-1) + traj_batch.reward.sum(axis=0).mean(axis=-1),
                        # [jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                        "total_rewards_per_game_estimate": (
                                                                       critic_train_state.shaped_reward_coeff * traj_batch.shaped_reward.sum(
                                                                   axis=0).mean(axis=-1) + traj_batch.reward.sum(
                                                                   axis=0).mean(axis=-1)) / (
                                                                       traj_batch.reward.shape[0] / 14),
                        "env_step": metric["update_steps"]
                                    * config["NUM_ENVS"]
                                    * config["NUM_STEPS"],
                        "shaped_coefficient": critic_train_state.shaped_reward_coeff,
                        "scaled_shaped_reward": critic_train_state.shaped_reward_coeff * traj_batch.shaped_reward.sum(
                            axis=0).mean(),
                        "final_reward": traj_batch.reward.sum(axis=0).mean(axis=-1),
                        # [jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                        "final_reward_per_game_estimate": traj_batch.reward.sum(axis=0).mean(axis=-1) / (
                                    traj_batch.reward.shape[0] // 14),
                        # [jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                        "advantages": advantages.sum(axis=0).mean(),
                        "targets": targets.sum(axis=0).mean(),
                        "total_loss": total_loss.sum(axis=0).mean(),
                        "actor_loss": actor_loss.sum(axis=0).mean(),
                        "critic_loss": critic_loss.sum(axis=0).mean(),
                        "entropy": entropy.sum(axis=0).mean()
                    }
                )
                # wandb.log(
                #     {
                #         "returns": metric["returned_episode_returns"][-1, :].mean(),
                #         "env_step": metric["update_steps"]
                #                     * config["NUM_ENVS"]
                #                     * config["NUM_STEPS"],
                #     }
                # )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric,critic_train_state,traj_batch,advantages, targets,total_loss,actor_loss,critic_loss,entropy)
            update_steps = update_steps + 1
            new_shaped_reward_coeff_value = 1.0
            # new_shaped_reward_coeff_value = jnp.maximum(0.0, 1 - (update_steps * config["NUM_ENVS"] * config["NUM_STEPS"] / 840000))# config["TOTAL_TIMESTEPS"]
            new_shaped_reward_coeff = jnp.full(
                critic_train_state.shaped_reward_coeff.shape, fill_value=new_shaped_reward_coeff_value)
            jax.debug.print("Shaped reward coeff: {a}, real_dsteps: {b}, shaped_reward_steps: {c}",
                            a=new_shaped_reward_coeff, b=update_steps * config["NUM_ENVS"] * config["NUM_STEPS"],
                            c=840000)
            # runner_state[1] is the training state object where the shaped reward coefficient is stored
            critic_train_state = critic_train_state.set_new_shaped_reward_coeff(
                new_coeff=new_shaped_reward_coeff)
            new_shaped_reward_coeff_actor = jnp.full(
                actor_train_state.shaped_reward_coeff.shape, fill_value=new_shaped_reward_coeff_value)
            # runner_state[1] is the training state object where the shaped reward coefficient is stored
            actor_train_state = actor_train_state.set_new_shaped_reward_coeff(
                new_coeff=new_shaped_reward_coeff_actor)
            train_states = (actor_train_state,critic_train_state)
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="mappo_homogenous_ff_yokai")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "FF", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)


if __name__ == "__main__":
    main()