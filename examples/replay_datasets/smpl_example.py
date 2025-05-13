from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment():
    env = ImitationFactory.make(
        env_name="UnitreeH1",
        # if SMPL and AMASS are installed, you can use the following:
        amass_dataset_conf=AMASSDatasetConf([
            "DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses",
            # "KIT/12/WalkInClockwiseCircle11_poses",
            # "HUMAN4D/HUMAN4D/Subject3_Medhi/INF_JumpingJack_S3_01_poses",
            # "KIT/359/walking_fast05_poses"
        ]),
        n_substeps=20,
    )

    env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)


if __name__ == "__main__":
    experiment()
