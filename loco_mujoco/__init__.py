from pathlib import Path

__version__ = '1.0.1'

try:
    PATH_TO_MODELS = Path(__file__).resolve().parent / "models"
    PATH_TO_VARIABLES = Path(__file__).resolve().parent / "LOCOMUJOCO_VARIABLES.yaml"
    PATH_TO_SMPL_ROBOT_CONF = Path(__file__).resolve().parent / "smpl" / "robot_confs"

    from .core import Mujoco, Mjx
    from .environments import LocoEnv
    from .task_factories import (TaskFactory, RLFactory, ImitationFactory)


    def get_registered_envs():
        """
        Returns a list of all registered environments in the LocoEnv class.

        This function provides access to the environments that have been registered
        within the LocoEnv class. It is useful for retrieving available environment
        names or identifiers dynamically during runtime.

        Args:
            None

        Returns:
            list[str]: A list containing the names or identifiers of all registered
                       environments.
        """
        return LocoEnv.registered_envs

except ImportError as e:
    print(e)
