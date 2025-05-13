import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment():
    """
    Performs an experiment using the ImitationFactory to create and interact with an environment.

    This function initializes an environment named "UnitreeH1" using the ImitationFactory.
    It configures the environment with a default dataset containing "squat" and "walk" actions
    and sets the number of simulation substeps to 20. The function then plays a trajectory
    in the environment for a specified number of episodes and steps, rendering the simulation
    for visualization.

    Parameters
    ----------
    env_name : str
        The name of the environment to be created. Fixed as "UnitreeH1".
    default_dataset_conf : DefaultDatasetConf
        Configuration object specifying the default dataset for the environment.
    n_substeps : int
        Number of simulation substeps per step in the environment.
    n_episodes : int
        Number of episodes to play in the environment.
    n_steps_per_episode : int
        Number of steps per episode during trajectory playback.
    render : bool
        Whether to render the environment during trajectory playback.

    Returns
    -------
    None
        This function does not return any value.
    """
    env = ImitationFactory.make(
        env_name="UnitreeH1",
        default_dataset_conf=DefaultDatasetConf(["squat", "walk"]),
        n_substeps=20,
    )

    env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)


if __name__ == "__main__":
    experiment()
