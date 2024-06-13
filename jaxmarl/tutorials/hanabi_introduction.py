from asyncio import start_server
import jax
from jaxmarl import make
from jaxmarl.environments.hanabi import hanabi
import time

# Parameters + random keys
max_steps = 100
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

env = make('hanabi')


obs, state = env.reset(key_r)
print('list of agents in environment', env.agents)

# Sample random actions
num_agent = 1024
rng, _rng = jax.random.split(key)
reset_rng = jax.random.split(_rng, num_agent)
start = time.time()
start_i = time.time()
obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
end = time.time()
print(end- start)
rng, _rng = jax.random.split(rng)
avail_actions = jax.vmap(env.get_legal_moves)(env_state)