import jax
from jaxmarl import make
import jax.numpy as jnp
import time
from jaxmarl.environments.yokai import Yokai
'''
hint:
can check with htop and nvtop in gpuws
TO test the env, can use unit test to provide some edge cases such as: if test the step_obs, then provide index in 0,8,15...
'''


# Parameters + random keys
max_steps = 100
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

env = make('yokai_game')


obs, state = env.reset(key_r)
key_a = jax.random.split(key_a, env.num_agents)
key, key_s, key_a = jax.random.split(key, 3)



for i in range(5):
    print('----------')
    num_envs = 100
    rng, _rng = jax.random.split(key)
    reset_rng = jax.random.split(_rng,num_envs)
    start = time.time()
    start_i = time.time()
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
    end = time.time()
    print(end-start)
    rng, _rng = jax.random.split(rng)
    start = time.time()
    avail_actions = jax.vmap(env.get_legal_moves)(env_state)
    end = time.time()
    print(end-start)
    rng_step = jax.random.split(_rng, num_envs)
    action_1 = jnp.array([2, 4, 143, 3])
    action_2 = jnp.array([3, 5, 393, 50])
    action_1_tiled = jnp.tile(action_1, (num_envs, 1))
    action_2_tiled = jnp.tile(action_2, (num_envs, 1))
    env_act = {'agent_0': action_1_tiled, 'agent_1': action_2_tiled}
    start = time.time()
    obsv_t, env_state_t, reward_t, done_t, info_t = jax.vmap(env.step, in_axes=(0, 0, 0))(
        rng_step, env_state, env_act
    )
    endi = time.time()
    print(endi-start)
    print('sum seconds')
    print(endi - start_i)
    print('finish')

'''
# action_terminate_1 = jnp.array([1])
# action_terminate_2 = jnp.array([1])
action_1 =jnp.array([2,4,143,3])
action_2 =jnp.array([3,5,393,50])#place 2 hint card on 10 yokai
action_3 =jnp.array([1,8,11,1])#move 0 to the left of 2 reveal 0 hint
action_4 =jnp.array([4,6,556,2])# move 8 to the right of 10,reveal 1 hint
action_5 =jnp.array([13,14,610,16])#move 9 to the bottom of 8 place hint 0 to 8
action_6 =jnp.array([4,6,778,33])# move 12 to the bottom of 2,place hint 1 tp 9
action_7 =jnp.array([13,14,406,4])#move 6 to the bottom of 5, reveal hint 3
action_8 =jnp.array([4,6,286,67])# move 4 to the bottom of 7,place hint 3 tp 11
action_9 =jnp.array([13,14,829,5])#move 12 to the upper of 15, reveal hint 4
action_10 =jnp.array([4,6,949,77])# move 14 to the upper of 13,place hint 4 tp 5
action_11 =jnp.array([2,4,9,6])#move 0 to the upper of 2, reveal hint 5
action_12 =jnp.array([2,4,11,100])# move 0 to the left 0f 2,place hint 5 tp 12
action_13 =jnp.array([2,4,9,7])#move 0 to the upper of 2, reveal hint 6
action_14 =jnp.array([2,4,11,118])# move 0 to the left 0f 2,place hint 6 tp 14
# action_space can be more concise, e.g. no-op in three steps can be change to 1 no-op
# actions_terminate = {'agent_0': action_terminate_1, 'agent_1': action_terminate_2}
# actions_terminate2 = {'agent_0': jnp.array([2]), 'agent_1': jnp.array([2])}
actions = {'agent_0': action_1, 'agent_1': action_2}
actions2 = {'agent_0': action_3, 'agent_1': action_4}
actions3 = {'agent_0': action_5, 'agent_1': action_6}
actions4 = {'agent_0': action_7, 'agent_1': action_8}
actions5 = {'agent_0': action_9, 'agent_1': action_10}
actions6 = {'agent_0': action_11, 'agent_1': action_12}
actions7 = {'agent_0': action_13, 'agent_1': action_14}
# actions_terminate2 = {'agent_0': action_terminate_1, 'agent_1': jnp.array([2])}
start = time.time()
obs1, state1, rewards1, dones1, infos1 = env.step(key_s, state, actions)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state1, actions)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions2)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions2)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions3)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions3)

obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions4)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions4)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions5)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions5)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions6)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions6)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions7)
obs2, state2, rewards2, dones2, infos2 = env.step(key_s, state2, actions7)
jax.debug.breakpoint()
legal= env.get_legal_moves(state2)


'''