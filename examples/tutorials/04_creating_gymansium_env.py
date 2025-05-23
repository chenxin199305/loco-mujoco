import numpy as np

"""
Gymnasium 是一个项目，为所有单智能体强化学习环境提供 API（应用程序编程接口），
并实现了常见环境：cartpole、pendulum、mountain-car、mujoco、atari 等。
本页将概述如何使用 Gymnasium 的基础知识，
包括其四个关键功能：make()、Env.reset()、Env.step() 和 Env.render()。
"""
import gymnasium as gym

import loco_mujoco
from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf

# Create the Gymnasium environment
env = gym.make(
    "LocoMujoco",
    env_name="SkeletonTorque",
    render_mode="human",
    default_dataset_conf=DefaultDatasetConf("walk"),
    lafan1_dataset_conf=LAFAN1DatasetConf("walk1_subject1"),
    goal_type="GoalTrajMimicv2",
    goal_params=dict(visualize_goal=True),
)

# Get the dimensionality of the action space
action_dim = env.action_space.shape[0]

# Set a random seed (currently unused)
seed = 1

# Reset the environment and initialize variables
env.reset()
img = env.render()

absorbing = False
i = 0

# Main simulation loop
while True:
    # Reset the environment after 1000 steps or if in an absorbing state
    if i == 1000 or absorbing:
        env.reset()
        i = 0

    # Sample a random action and apply it to the environment
    action = np.random.randn(action_dim)
    nstate, _, absorbing, _, _ = env.step(action)

    # Render the environment
    env.render()
    i += 1
