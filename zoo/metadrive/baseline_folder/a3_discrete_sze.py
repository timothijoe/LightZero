from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = False
K = 20  # num_of_sampled_actions
collector_env_num = 14
n_episode = 14
evaluator_env_num = 14
num_simulations = 100
update_per_collect = 200
batch_size = 64
max_env_step = int(1e7)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_sampled_efficientzero_config = dict(
    exp_name=
    f'data_icra_sep9_ctree/compare3_2_discrete_sez_{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_name='Pendulum-v1',
        continuous=True,
        obs_shape = [5, 200, 200],
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        metadrive=dict(
            use_render=False,
            show_seq_traj = False,
            traffic_density = 0.3,
            seq_traj_len = 1,
            traj_control_mode = 'acc',
            avg_speed = 5.0,
            use_lateral=True,
            use_speed_reward = True,
            use_heading_reward = True,
            use_jerk_reward = True,
            heading_reward=0.15,
            speed_reward = 0.00,
            driving_reward = 0.2,
            ignore_first_steer = False,
            crash_vehicle_penalty = 4.0,
            out_of_road_penalty = 5.0,
            is_continuous = False,
            jerk_importance=0.1,
        ),
    ),
    policy=dict(
        model=dict(
            observation_shape=[5, 200, 200],
            action_space_size=49,
            lstm_hidden_size=128,
            latent_state_dim=128,
            downsample = True,
            image_channel=5,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
        ),
        cuda=True,
        env_type='not_board_games',
        threshold_training_steps_for_final_temperature = 20000,
        threshold_training_steps_for_final_lr = 20000,
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=True,
        manual_temperature_decay=True,
        learning_rate=0.0003,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(3e4),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        discount_factor=0.99,
        td_steps = 2,
        num_unroll_steps = 2,
        normalize_prob_of_sampled_actions = True,
        learn=dict(
            learner=dict(
                hook=dict(
                    save_ckpt_after_iter=5000000,
                ),
            ),
        ),
    ),
)
pendulum_sampled_efficientzero_config = EasyDict(pendulum_sampled_efficientzero_config)
main_config = pendulum_sampled_efficientzero_config

pendulum_sampled_efficientzero_create_config = dict(
    env=dict(
        type='pendulum_lightzero',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
pendulum_sampled_efficientzero_create_config = EasyDict(pendulum_sampled_efficientzero_create_config)
create_config = pendulum_sampled_efficientzero_create_config


if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        from lzero.entry.train_metadrive import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        from lzero.entry.train_metadrive import train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)