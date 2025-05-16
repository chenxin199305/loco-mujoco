import os
import jax
import time
from loco_mujoco import ImitationFactory

# Optimize GPU performance
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

# Create the environment
env = ImitationFactory.make(
    "MjxUnitreeG1",
    default_dataset_conf=dict(task="stepinplace1")
)

# Initialize random keys
rng_key = jax.random.key(0)
n_envs = 100
rng_keys = jax.random.split(rng_key, n_envs + 1)
rng_key, env_keys = rng_keys[0], rng_keys[1:]

# JIT-compile and vectorize environment functions
"""
In JAX and reinforcement learning (RL), random number generators (RNGs) are essential for several reasons, especially when combined with jax.jit and jax.vmap for parallelized environments. Here's why they're needed in your code:
1. Reproducibility & Deterministic Control

JAX requires explicit RNGs to ensure deterministic randomness, unlike NumPy/PyTorch, which rely on global state.

    In RL, you need randomness for:

        Environment resets (e.g., random initial states).

        Action sampling (e.g., exploration in RL).

        Stochastic transitions (if the environment has randomness).

    By passing RNGs explicitly, you avoid hidden state and ensure reproducibility.

Example:
python

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
random_action = rng_sample_uni_action(subkey)  # Deterministic!

2. Parallelization with vmap Requires Split RNGs

When using jax.vmap to run multiple environments in parallel:

    Each environment needs its own independent randomness (e.g., different initial states, different action noise).

    JAX’s RNGs must be split to avoid duplicate randomness across batches.

Example:
python

batch_size = 8
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, batch_size)  # Split for parallel envs
reset_states = rng_reset(keys)           # Each env gets unique randomness

If you don’t split keys properly, all parallel environments would get the same "random" values (breaking independence).
3. JIT Compilation Constraints

jax.jit compiles functions statically, meaning:

    No Python-side randomness (e.g., np.random) works inside JIT.

    RNGs must be passed as arguments (not created inside JIT’d functions).

Your functions (rng_reset, rng_step, rng_sample_uni_action) are JIT-compiled, so they must take RNGs as inputs.
4. Reinforcement Learning-Specific Needs

    Exploration: Agents (e.g., in DQN, PPO) need randomness to explore (e.g., ε-greedy, Gaussian noise).

    Environment Stochasticity: Some environments (e.g., CartPole with random pushes) need RNGs for transitions.

    Batch Initialization: Parallel training requires different random seeds per environment.
    
核心原因还是 JIT 编译和 vmap 需要 RNGs 来确保每个环境的独立性和可重复性。JAX 的 RNG 系统设计是为了支持高效的并行计算和可预测的随机性。（参见函数式编程）
"""
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

# Reset the environment
state = rng_reset(env_keys)

# Simulation parameters
step = 0
LOGGING_FREQUENCY = 100000
previous_time = time.time()

# Simulation loop
for i in range(100000):
    # Sample actions and step the environment
    rng_keys = jax.random.split(rng_key, n_envs + 1)
    rng_key, action_keys = rng_keys[0], rng_keys[1:]

    # Action here are sampled from a uniform distribution
    action = rng_sample_uni_action(action_keys)
    state = rng_step(state, action)

    # Render the environment
    env.mjx_render(state)

    # Log simulation speed
    step += n_envs
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        steps_per_second = int(LOGGING_FREQUENCY / (current_time - previous_time))
        print(f"{steps_per_second} steps per second.")
        previous_time = current_time
