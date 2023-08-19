from easydict import EasyDict
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
TRAJ_CONTROL_MODE = 'acc'
SEQ_TRAJ_LEN = 10

continuous_action_space = True
K = 5  # num_of_sampled_actions
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 20
update_per_collect = 5
batch_size = 256 #256
max_env_step = int(1e6)
reanalyze_ratio = 0.0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez_ctree/pendulum_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
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
            seq_traj_len = SEQ_TRAJ_LEN,
            traj_control_mode = TRAJ_CONTROL_MODE,
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
            use_cross_line_penalty=True,
            use_explicit_vel_obs = False,
        ),
    ),
    policy=dict(
        model=dict(
            observation_shape=[5, 200, 200],
            action_space_size=3,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            model_type='conv',  # options={'mlp', 'conv'}
            lstm_hidden_size=128,
            latent_state_dim=128,
            downsample = True,
            image_channel=6,
        ),
        # learn=dict(
        #     learner=dict(
        #         hook=dict(
        #             load_ckpt_before_run='/home/PJLAB/puyuan/jiqun_data/gmm_mcts_best.pth.tar',
        #         ),
        #     ),
        # ),        
        cuda=True,
        env_type='not_board_games',
        mcts_ctree=False,
        use_expert= True,
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(10),
        replay_buffer_size=int(64),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
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


# if __name__ == "__main__":
#     # Users can use different train entry by specifying the entry_type.
#     entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

#     if entry_type == "train_muzero":
#         from lzero.entry.train_metadrive import train_muzero
#     elif entry_type == "train_muzero_with_gym_env":
#         from lzero.entry.train_metadrive import train_muzero

#     train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
    
if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        # from lzero.entry import train_muzero
        from lzero.entry.eval_metadrive import eval_metadrive
    zt_path = '/home/hunter/obelisk/data_ckpt/f_server/ckpt_best_v65.pth.tar'

    eval_metadrive([main_config, create_config], seed=0, model_path=zt_path,num_episodes_each_seed=5)
