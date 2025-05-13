from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment():
    env = ImitationFactory.make(
        env_name="UnitreeH1",
        lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
        n_substeps=20,
    )

    env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)


if __name__ == "__main__":
    experiment()
