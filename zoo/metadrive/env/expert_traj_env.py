import os
import copy
import time
import gym
import numpy as np
from gym import spaces
from collections import defaultdict
from typing import Union, Dict, AnyStr, Tuple, Optional
from gym.envs.registration import register
import logging

from zoo.metadrive.utils.discrete_policy import DiscreteMetaAction
# from zoo.metadrive.utils.agent_manager_utils import MacroAgentManager
from zoo.metadrive.expert_utils.expert_agent_manager_utils import ExpertAgentManager as MacroAgentManager
from zoo.metadrive.utils.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed, MacroBaseEngine
from zoo.metadrive.utils.traffic_manager_utils import TrafficMode
from zoo.metadrive.utils.observation_utils import HRLTopDownMultiChannel
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT, REPLAY_DONE
from metadrive.envs.base_env import BaseEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
# from metadrive.manager.traffic_manager import TrafficMode
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils import Config, merge_dicts, get_np_random, clip
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.utils.utils import auto_termination
# from core.policy.ad_policy.traj_vae import VaeDecoder
import torch
from metadrive.component.road_network import Road
from zoo.metadrive.utils.traj_decoder import VaeDecoder

vae_load_dir = 'zoo/metadrive/model/nov02_len10_dim3_v1_ckpt'
vae_load_dir = '/home/hunter/hoffung/LightZero/' + vae_load_dir 
_traj_decoder = VaeDecoder(
            embedding_dim = 64,
            h_dim = 64,
            latent_dim = 3,
            seq_len = 10,
            dt = 0.1,
            traj_control_mode = 'acc',
            one_side_class_vae=False,
            steer_rate_constrain_value=0.5,
        )
_traj_decoder.load_state_dict(torch.load(vae_load_dir,map_location=torch.device('cpu')))

def convert_wp_to_world_coord3d(wp, robot_pos):
    odom_goal = [0.0, 0.0, 0.0]
    local_goal = wp
    delta_length = np.sqrt(local_goal[0] ** 2 + local_goal[1] ** 2)
    delta_angle = np.arctan2(local_goal[1], local_goal[0])
    total_angle = delta_angle + robot_pos[2]
    odom_goal[0] = delta_length * np.cos(total_angle) + robot_pos[0]
    odom_goal[1] = delta_length * np.sin(total_angle) + robot_pos[1]
    yaw_z = local_goal[2] + robot_pos[2]
    odom_goal[2] = np.arctan2(np.sin(yaw_z), np.cos(yaw_z))
    return odom_goal   

def convert_wp_to_world_coord(wp, robot_pos):
    odom_goal = [0.0, 0.0]
    local_goal = wp
    delta_length = np.sqrt(local_goal[0] ** 2 + local_goal[1] ** 2)
    delta_angle = np.arctan2(local_goal[1], local_goal[0])
    total_angle = delta_angle + robot_pos[2]
    odom_goal[0] = delta_length * np.cos(total_angle) + robot_pos[0]
    odom_goal[1] = delta_length * np.sin(total_angle) + robot_pos[1]
    return odom_goal        

def convert_waypoint_list_coord(wp_list, rbt_pos):
    # given the robot initial pos, we transfer the traj from origin to initial pos, and rotation
    wp_w_list = []
    for wp in wp_list:
        wp_w = convert_wp_to_world_coord3d(wp, rbt_pos)
        wp_w_list.append(wp_w)
    return wp_w_list

def process_node(starting_state, latent_action):
    child_num = len(latent_action)
    latent_action = np.array(latent_action)
    starting_state = starting_state.reshape(1, -1)
    starting_state = np.repeat(starting_state, child_num, axis = 0)
    starting_state_taec = copy.deepcopy(starting_state)
    starting_state_taec[:, :3] = 0
    latent_action_torch = torch.from_numpy(latent_action).to(torch.float32)
    starting_state_torch = torch.from_numpy(starting_state_taec).to(torch.float32)
    with torch.no_grad():
        traj = _traj_decoder(latent_action_torch, starting_state_torch)
    traj = traj.numpy()
    starting_point = starting_state_taec[:,:4]
    starting_point = np.expand_dims(starting_point, axis = 1)
    traj = np.concatenate((starting_point, traj), axis=1)
    convert_traj_list = []
    
    for i in range(child_num):
        single_starting = starting_state[i][:3]
        single_traj = traj[i]
        single_traj_3 = single_traj[:,:3]
        single_traj_v = single_traj[:, 3:4]
        single_traj_3 = list(single_traj_3)
        convert_traj_3 = convert_waypoint_list_coord(single_traj_3, single_starting)
        convert_traj_3 = np.array(convert_traj_3)
        convert_traj_4 = np.concatenate((convert_traj_3, single_traj_v), axis=1)
        convert_traj_list.append(convert_traj_4)
    convert_traj_array = np.array(convert_traj_list)
    return convert_traj_array 

def traverse_dict(node_dict, starting_state, total_traj = []):
    node_dict["starting_pose"] = starting_state[:4]
    print('node id: {}'.format(node_dict["node_id"]))
    # root node starting state ([x, y, theta, v]) w.r.t current car position
    children = node_dict["children"]
    if children is None:
        return
    child_latent_actions = []
    index_fun = {}
    index = 0
    for child_dict in children:
        child_latent_actions.append(child_dict["motivation"])
        index_fun[index] = child_dict["node_id"]
    traj_list = process_node(starting_state, child_latent_actions)
    child_num = traj_list.shape[0]
    for i in range(child_num):
        traj_i = traj_list[i]
        total_traj.append(traj_i)
        starting_state_i = traj_i[-1]
        traverse_dict(children[i], starting_state_i, total_traj)
        
    # process_parent_node(node_dict)
    # children = node_dict["children"]
    # for child_dict in children:
    #     # 添加新键
    #     child_dict["new_key"] = "new_value"
    #     traverse_dict(child_dict)
    #     # 处理子节点
    #     process_child_node(child_dict)


DIDRIVE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    use_render=False,
    environment_num=10,

    # ===== Map Config =====
    map='SSSSSSSSSS',  # int or string: an easy way to fill map_config
    # map='SSSSSSS',
    random_lane_width=True,
    random_lane_num=True,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        BaseMap.GENERATE_CONFIG: 'SSSSSSSSSS',  # None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 70,
    },

    # ===== Traffic =====
    traffic_density=0.0,
    on_screen=False,
    rgb_clip=True,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Synch,  # "Respawn", "Trigger"
    random_traffic=True,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,
    is_multi_agent=False,
    vehicle_config=dict(spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)),

    # ===== Agent =====
    target_vehicle_configs={
        DEFAULT_AGENT: dict(use_special_color=True, spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 2))
    },

    # ===== Reward Scheme =====
    # See: https://github.com/decisionforce/metadrive/issues/283
    success_reward= 10.0, #10.0,
    out_of_road_penalty= 1.0, #1.0,
    crash_vehicle_penalty=1.0, #1.0,
    crash_object_penalty=5.0, #5.0,
    run_out_of_time_penalty = 5.0, #5.0,
    driving_reward=0.1,
    speed_reward=0.2,
    heading_reward = 0.3, 


    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=True,
    #physics_world_step_size=1e-1,
    physics_world_step_size=0.1,

    # ===== Trajectory length =====
    seq_traj_len = 10,
    show_seq_traj = False,
    debug_info=False,
    enable_u_turn = False,
    #episode_max_step = 100,
    
    zt_mcts = True,
    expert_type = None, # 1, 2, 3

    


    #traj_control_mode = 'acc', # another type is 'jerk'
    traj_control_mode = 'jerk',
    # if we choose traj_control_mode = 'acc', then the current state is [0,0,0,v] and the control signal is throttle and steer
    # If not, we will use jerk control, the current state we have vel, acc, current steer, and the control signal is jerk and steer rate (delta_steer)
    
    # Reward Option Scheme
    const_episode_max_step = False,
    episode_max_step = 150,
    avg_speed = 6.5,

    use_lateral=True,
    lateral_scale = 0.25, 

    jerk_bias = 15.0, 
    jerk_dominator = 45.0, #50.0
    jerk_importance = 0.6, # 0.6
    use_speed_reward = True,
    use_heading_reward = False,
    use_jerk_reward = False,
    ignore_first_steer = False,
    add_extra_speed_penalty = False,
    use_cross_line_penalty = False,
    use_explicit_vel_obs = False,
    use_explicit_vel_obs_compare = False,
)


class MetaDriveTrajEnv(BaseEnv):

    @classmethod
    def default_config(cls) -> "Config":
        #config = super(SimpleMetaDriveEnv, cls).default_config()
        config = Config(BASE_DEFAULT_CONFIG)
        config.update(DIDRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: dict = None):
        merged_config = self._merge_extra_config(config)
        global_config = self._post_process_config(merged_config)
        self.config = global_config

        # if self.config["seq_traj_len"] == 1:
        #     self.config["episode_max_step"] = self.config["episode_max_step"] * 10
        # if self.config["seq_traj_len"] == 20:
        #     self.config["episode_max_step"] = self.config["episode_max_step"] // 2

        # agent check
        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1
        assert isinstance(self.num_agents, int) and (self.num_agents > 0 or self.num_agents == -1)

        # observation and action space
        self.agent_manager = MacroAgentManager(
            init_observations=self._get_observations(), init_action_space=self._get_action_space()
        )
        self.action_type = DiscreteMetaAction()
        
        
    
        
        #self.action_space = self.action_type.space()

        # lazy initialization, create the main vehicle in the lazy_init() func
        self.engine: Optional[MacroBaseEngine] = None
        self._top_down_renderer = None
        self.episode_steps = 0
        # self.current_seed = None

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.dones = None
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        self.start_seed = self.config["start_seed"]
        self.env_num = self.config["environment_num"]

        self.time = 0
        self.step_num = 0
        self.episode_rwd = 0
        # self.vae_decoder = VaeDecoder(
        #         embedding_dim = 64,
        #         h_dim = 64,
        #         latent_dim = 2,
        #         seq_len = self.config['seq_traj_len'],
        #         dt = 0.1
        #     )
        # # vae_load_dir = 'ckpt_files/a79_decoder_ckpt'
        # # vae_load_dir = '/home/SENSETIME/zhoutong/hoffnung/xad/ckpt_files/seq_len_20_79_decoder_ckpt'
        # if self.config['seq_traj_len'] == 10:
        #     vae_load_dir = 'ckpt_files/seq_len_10_decoder_ckpt'
        # elif self.config['seq_traj_len'] == 15:
        #     vae_load_dir = 'ckpt_files/seq_len_15_78_decoder_ckpt'
        # else:
        #     assert self.config['seq_traj_len'] == 20
        #     vae_load_dir = 'ckpt_files/seq_len_20_79_decoder_ckpt'
        # self.vae_decoder.load_state_dict(torch.load(vae_load_dir))
        self.vel_speed = 0.0
        self.z_state = np.zeros(6)
        self.z_xyt = np.zeros(3)
        self.avg_speed = self.config["avg_speed"]
        vae_load_dir = 'zoo/metadrive/model/nov02_len10_dim3_v1_ckpt'
        vae_load_dir = '/home/hunter/hoffung/LightZero/' + vae_load_dir 
        self._traj_decoder = VaeDecoder(
            embedding_dim = 64,
            h_dim = 64,
            latent_dim = 3,
            seq_len = 10,
            dt = 0.1,
            traj_control_mode = 'acc',
            one_side_class_vae=False,
            steer_rate_constrain_value=0.5,
        )
        # self._traj_decoder.load_state_dict(torch.load(vae_load_dir))
        self._traj_decoder.load_state_dict(torch.load(vae_load_dir,map_location=torch.device('cpu')))

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 1, shape=(200, 200, 5), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, shape=(3, ), dtype=np.float32)

    @property
    def reward_space(self):
        return gym.spaces.Box(-100, 100, shape=(1, ), dtype=np.float32)    
    # define a action type, and execution style
    # Now only one action will be taken, cosin function, and we set dt equals self.engine.dt
    # now that in this situation, we directly set trajectory len equals to simulation frequency

    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        self.episode_steps += 1
        macro_actions = self._preprocess_macro_waypoints(actions)
        time_1 = time.time() 
        step_infos = self._step_macro_simulator(macro_actions)
        time_2 = time.time() 
        o, r, d, i = self._get_step_return(actions, step_infos)
        time_3 = time.time() 
        self.step_num = self.step_num + 1
        self.episode_rwd = self.episode_rwd + r 
        #print('step number is: {}'.format(self.step_num))
        #o = o.transpose((2,0,1))
        return o, r, d, i

    def get_waypoint_list(self):
        x = np.arange(0, 6.2, 0.2)
        LENGTH = 0 # 4.51
        y = 1 * np.cos(np.pi*2 / 6.0 * x)-1
        x = x + LENGTH/2
        lst = []
        for i in range(x.shape[0]):
            lst.append([x[i],y[i]])
        return lst

    def _merge_extra_config(self, config: Union[dict, "Config"]) -> "Config":
        config = self.default_config().update(config, allow_add_new_key=True)
        if config["vehicle_config"]["lidar"]["distance"] > 50:
            config["max_distance"] = config["vehicle_config"]["lidar"]["distance"]
        return config

    def _post_process_config(self, config):
        config = super(MetaDriveTrajEnv, self)._post_process_config(config)
        if not config["rgb_clip"]:
            logging.warning(
                "You have set rgb_clip = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )
        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config()
        )
        config["vehicle_config"]["rgb_clip"] = config["rgb_clip"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        if config.get("gaussian_noise", 0) > 0:
            assert config["vehicle_config"]["lidar"]["gaussian_noise"] == 0, "You already provide config!"
            assert config["vehicle_config"]["side_detector"]["gaussian_noise"] == 0, "You already provide config!"
            assert config["vehicle_config"]["lane_line_detector"]["gaussian_noise"] == 0, "You already provide config!"
            config["vehicle_config"]["lidar"]["gaussian_noise"] = config["gaussian_noise"]
            config["vehicle_config"]["side_detector"]["gaussian_noise"] = config["gaussian_noise"]
            config["vehicle_config"]["lane_line_detector"]["gaussian_noise"] = config["gaussian_noise"]
        if config.get("dropout_prob", 0) > 0:
            assert config["vehicle_config"]["lidar"]["dropout_prob"] == 0, "You already provide config!"
            assert config["vehicle_config"]["side_detector"]["dropout_prob"] == 0, "You already provide config!"
            assert config["vehicle_config"]["lane_line_detector"]["dropout_prob"] == 0, "You already provide config!"
            config["vehicle_config"]["lidar"]["dropout_prob"] = config["dropout_prob"]
            config["vehicle_config"]["side_detector"]["dropout_prob"] = config["dropout_prob"]
            config["vehicle_config"]["lane_line_detector"]["dropout_prob"] = config["dropout_prob"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["target_vehicle_configs"][DEFAULT_AGENT])
            config["target_vehicle_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def _get_observations(self):
        return {DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def done_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        done = False
        done_info = dict(
            crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False
        )
        if vehicle.arrive_destination:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        elif hasattr(vehicle, 'macro_succ') and vehicle.macro_succ:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        elif hasattr(vehicle, 'macro_crash') and vehicle.macro_crash:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        if self._is_out_of_road(vehicle):
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info[TerminationState.OUT_OF_ROAD] = True
        if vehicle.crash_vehicle:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        if vehicle.crash_object:
            done = True
            done_info[TerminationState.CRASH_OBJECT] = True
            logging.info("Episode ended! Reason: crash object ")
        if vehicle.crash_building:
            done = True
            done_info[TerminationState.CRASH_BUILDING] = True
            logging.info("Episode ended! Reason: crash building ")
        if self.step_num >= self.episode_max_step:
            done = True
            done_info[TerminationState.CRASH_BUILDING] = True
            logging.info("Episode ended! Reason: crash building ")

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING]
        )
        done_info['complete_ratio'] = clip(self.already_go_dist/ self.navi_distance + 0.05, 0.0, 1.0)
        done_info['seq_traj_len'] = self.config['seq_traj_len']

        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        elif self.step_num > self.config["episode_max_step"]:
            step_info['cost'] = 1
        return step_info['cost'], step_info

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or \
              (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        if self._compute_navi_dist:
            self.navi_distance = self.get_navigation_len(vehicle)
            if not self.config['const_episode_max_step']:
                self.episode_max_step = self.get_episode_max_step(self.navi_distance, self.avg_speed)
            self._compute_navi_dist = False
        #self.update_current_state(vehicle)
        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_macro_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        self.already_go_dist += (long_now - long_last)
        lateral_factor = 1.0
        #     use_lateral_penalty = True
        reward = 0.0
        step_info["step_reward"] = reward 
        append_rwd = 0.4 * (self.episode_max_step - self.step_num)
        if append_rwd < 0.0:
            append_rwd = 0.0
        if vehicle.arrive_destination:
            reward = +self.config["success_reward"]
            reward += append_rwd
        elif vehicle.macro_succ:
            reward = +self.config["success_reward"]
            reward += append_rwd
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.macro_crash:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif self.step_num >= self.episode_max_step:
            reward = - self.config["run_out_of_time_penalty"]
        return reward, step_info
    
    def get_navigation_len(self, vehicle):
        checkpoints = vehicle.navigation.checkpoints
        road_network = vehicle.navigation.map.road_network
        total_dist = 0
        assert len(checkpoints) >=2
        for check_num in range(0, len(checkpoints)-1):
            front_node = checkpoints[check_num]
            end_node = checkpoints[check_num+1] 
            cur_lanes = road_network.graph[front_node][end_node]
            target_lane_num = int(len(cur_lanes) / 2)
            target_lane = cur_lanes[target_lane_num]
            target_lane_length = target_lane.length
            total_dist += target_lane_length 

        if hasattr(vehicle.navigation, 'u_turn_case'):
            if vehicle.navigation.u_turn_case is True:
                total_dist += 35
        return total_dist
            
    def compute_jerk_list(self, vehicle):
        jerk_list = []
        #vehicle = self.vehicles[vehicle_id]
        v_t0 = vehicle.penultimate_state['speed']
        theta_t0 = vehicle.penultimate_state['yaw']
        v_t1 = vehicle.traj_wp_list[0]['speed']
        theta_t1 = vehicle.traj_wp_list[0]['yaw']
        v_t2 = vehicle.traj_wp_list[1]['speed']
        theta_t2 = vehicle.traj_wp_list[1]['yaw']
        t_inverse = 1.0 / self.config['physics_world_step_size']
        first_point_jerk_x = (v_t2* np.cos(theta_t2) - 2 * v_t1 * np.cos(theta_t1) +  v_t0 * np.cos(theta_t0)) * t_inverse * t_inverse
        first_point_jerk_y = (v_t2* np.sin(theta_t2) - 2 * v_t1 * np.sin(theta_t1) +  v_t0 * np.sin(theta_t0)) * t_inverse * t_inverse
        jerk_list.append(np.array([first_point_jerk_x, first_point_jerk_y]))
        # plus one because we store the current value as first, which means the whole trajectory is seq_traj_len + 1
        for i in range(2, self.config['seq_traj_len'] + 1):
            v_t0 = vehicle.traj_wp_list[i-2]['speed']
            theta_t0 = vehicle.traj_wp_list[i-2]['yaw']
            v_t1 = vehicle.traj_wp_list[i-1]['speed']
            theta_t1 = vehicle.traj_wp_list[i-1]['yaw']
            v_t2 = vehicle.traj_wp_list[i]['speed']
            theta_t2 = vehicle.traj_wp_list[i]['yaw']    
            point_jerk_x = (v_t2* np.cos(theta_t2) - 2 * v_t1 * np.cos(theta_t1) + v_t0 * np.cos(theta_t0)) * t_inverse * t_inverse
            point_jerk_y = (v_t2* np.sin(theta_t2) - 2 * v_t1 * np.sin(theta_t1) + v_t0 * np.sin(theta_t0)) * t_inverse * t_inverse
            jerk_list.append(np.array([point_jerk_x, point_jerk_y]))
        #final_jerk_value = 0
        step_jerk_list = []
        for jerk in jerk_list:
            #final_jerk_value += np.linalg.norm(jerk)
            step_jerk_list.append(np.linalg.norm(jerk))
        return step_jerk_list


    # def update_current_state(self, vehicle):
    #     vehicle = self.vehicles[vehicle_id]
    #     t_inverse = 1.0 / self.config['physics_world_step_size']
    #     theta_t1 = vehicle.traj_wp_list[-2]['yaw']
    #     theta_t2 = vehicle.traj_wp_list[-1]['yaw']
    #     v_t1 = vehicle.traj_wp_list[-2]['speed']
    #     v_t2 = vehicle.traj_wp_list[-1]['speed']
    #     v_state = np.zeros(6)
    #     v_state[3] = v_t2
    #     v_state[4] = (v_t2 - v_t1) * t_inverse 
    #     theta_dot = (theta_t2 - theta_t1) * t_inverse
    #     v_state[5] = np.arctan(2.5 * theta_dot / v_t2) if v_t2 > 0.001 else 0.0
    #     self.z_state = v_state

    def update_current_state(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        t_inverse = 1.0 / self.config['physics_world_step_size']
        theta_t1 = vehicle.traj_wp_list[-2]['yaw']
        theta_t2 = vehicle.traj_wp_list[-1]['yaw']
        v_t1 = vehicle.traj_wp_list[-2]['speed']
        v_t2 = vehicle.traj_wp_list[-1]['speed']
        v_state = np.zeros(6)
        v_state[3] = v_t2
        v_state[4] = (v_t2 - v_t1) * t_inverse 
        theta_dot = (theta_t2 - theta_t1) * t_inverse
        v_state[5] = np.arctan(2.5 * theta_dot / v_t2) if v_t2 > 0.001 else 0.0
        self.z_state = v_state
        xyt = np.zeros(3)
        xyt[0] = vehicle.traj_wp_list[-1]['position'][0]
        xyt[1] = vehicle.traj_wp_list[-1]['position'][1]
        xyt[2] = vehicle.traj_wp_list[-1]['yaw']
        self.z_xyt = xyt
        if hasattr(vehicle, 'vis_state'):
            vehicle.vis_state = copy.deepcopy(self.z_state)

    def compute_heading_error_list(self, vehicle, lane):
        heading_error_list = []
        for i in range(1, self.config['seq_traj_len'] + 1):
            theta = vehicle.traj_wp_list[i]['yaw'] 
            long_now, lateral_now = lane.local_coordinates(vehicle.traj_wp_list[i]['position'])
            road_heading_theta = lane.heading_theta_at(long_now)
            theta_error = self.wrap_angle(theta - road_heading_theta)
            heading_error_list.append(np.abs(theta_error))
        return heading_error_list

    def compute_speed_list(self, vehicle):
        speed_list = []
        for i in range(1, self.config['seq_traj_len'] + 1):
            speed = vehicle.traj_wp_list[i]['speed']
            speed_list.append(speed)
        return speed_list

    def compute_avg_lateral_cum(self, vehicle, lane):
        # Compute lateral distance for each wp
        # average the factor by seq traj len
        # For example, if traj len is 10, then i = 1, 2, ... 10
        lateral_cum = 0
        for i in range(1, self.config['seq_traj_len'] + 1):
            long_now, lateral_now = lane.local_coordinates(vehicle.traj_wp_list[i]['position'])
            lateral_cum += np.abs(lateral_now)
        avg_lateral_cum = lateral_cum / float(self.config['seq_traj_len'])
        return avg_lateral_cum

    def switch_to_third_person_view(self) -> None:
        if self.main_camera is None:
            return
        self.main_camera.reset()
        if self.config["prefer_track_agent"] is not None and self.config["prefer_track_agent"] in self.vehicles.keys():
            new_v = self.vehicles[self.config["prefer_track_agent"]]
            current_track_vehicle = new_v
        else:
            if self.main_camera.is_bird_view_camera():
                current_track_vehicle = self.current_track_vehicle
            else:
                vehicles = list(self.engine.agents.values())
                if len(vehicles) <= 1:
                    return
                if self.current_track_vehicle in vehicles:
                    vehicles.remove(self.current_track_vehicle)
                new_v = get_np_random().choice(vehicles)
                current_track_vehicle = new_v
        self.main_camera.track(current_track_vehicle)
        return

    def switch_to_top_down_view(self):
        self.main_camera.stop_track()

    def _get_step_return(self, actions, engine_info):
        # update obs, dones, rewards, costs, calculate done at first !
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.vehicles.items():
            o = self.observations[v_id].observe(v)
            self.update_current_state(v_id)
            self.vel_speed = v.last_spd
            if self.config["traj_control_mode"] == 'jerk':
                o_dict = {}
                o_dict['birdview'] = o 
                # v_state = np.zeros(4)
                # v_state[3] = v.last_spd
                v_state = self.z_state
                o_dict['vehicle_state'] = v_state
                #o_dict['speed'] = v.last_spd
            elif self.config["traj_control_mode"] == 'acc':
                o_dict = {}
                o_dict['birdview'] = o 
                # v_state = np.zeros(4)
                # v_state[3] = v.last_spd
                v_state = self.z_state[:6] #:4
                if self.config['ignore_first_steer']:
                    v_state[5] = 0.0
                # v_state[3] = 5.0
                o_dict['vehicle_state'] = v_state
                #o_dict['speed'] = v.last_spd
            else:
                o_dict = o
            obses[v_id] = o_dict

            done_function_result, done_infos[v_id] = self.done_function(v_id)
            rewards[v_id], reward_infos[v_id] = self.reward_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)
            done = done_function_result or self.dones[v_id]
            if self.config['use_explicit_vel_obs']:
                max_spd = 10.0
                cur_spd = self.z_state[3]
                a_pixel = cur_spd / max_spd
                a_pixel = clip(a_pixel, 0.0, 1.0)
                max_steer = 1.0
                cur_steer = self.z_state[5] + max_steer 
                b_pixel = cur_steer / max_steer 
                b_pixel = clip(b_pixel, 0.0, 1.0)
                additional_channel = np.zeros((200, 200, 1))
                original_channels = obses['default_agent']['birdview']
                combined_channel = np.concatenate((original_channels, additional_channel), axis = 2)
                obses['default_agent']['birdview'] = combined_channel
                obses['default_agent']['birdview'][:,:, 5]=0.0
                obses['default_agent']['birdview'][:,0:50, 5]=a_pixel
                obses['default_agent']['birdview'][:,50:90, 5]=b_pixel
                if self.config['use_explicit_vel_obs_compare']:
                    obses['default_agent']['birdview'][:,:, 5]=0.0
            self.dones[v_id] = done
            if done:
                if self.config['use_explicit_vel_obs']:
                    obses['default_agent']['birdview'][:,100:200, 5]=1.0
                else:
                    obses['default_agent']['birdview'][:,100:200, 4]=1.0

        should_done = engine_info.get(REPLAY_DONE, False
                                      ) or (self.config["horizon"] and self.episode_steps >= self.config["horizon"])
        termination_infos = self.for_each_vehicle(auto_termination, should_done)

        step_infos = concat_step_infos([
            engine_info,
            done_infos,
            reward_infos,
            cost_infos,
            termination_infos,
        ])

        if should_done:
            for k in self.dones:
                self.dones[k] = True

        dones = {k: self.dones[k] for k in self.vehicles.keys()}
        for v_id, r in rewards.items():
            self.episode_rewards[v_id] += r
            step_infos[v_id]["episode_reward"] = self.episode_rewards[v_id]
            self.episode_lengths[v_id] += 1
            step_infos[v_id]["episode_length"] = self.episode_lengths[v_id]
        if not self.is_multi_agent:
            return self._wrap_as_single_agent(obses), self._wrap_as_single_agent(rewards), \
                   self._wrap_as_single_agent(dones), self._wrap_as_single_agent(step_infos)
        else:
            return obses, rewards, dones, step_infos

    def setup_engine(self):
        super(MetaDriveTrajEnv, self).setup_engine()
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("q", self.switch_to_third_person_view)
        from zoo.metadrive.utils.traffic_manager_utils import MacroTrafficManager
        from zoo.metadrive.utils.map_manager_utils import MacroMapManager
        self.engine.register_manager("map_manager", MacroMapManager())
        self.engine.register_manager("traffic_manager", MacroTrafficManager())

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else \
            get_np_random(self._DEBUG_RANDOM_SEED).randint(self.start_seed, self.start_seed + self.env_num)
        self.seed(current_seed)

    def _preprocess_macro_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray]]:
        if not self.is_multi_agent:
            # print('action.dtype: {}'.format(type(actions)))
            #print('action: {}'.format(actions))
            actions = int(actions)
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        else:
            if self.config["vehicle_config"]["action_check"]:
                # Check whether some actions are not provided.
                given_keys = set(actions.keys())
                have_keys = set(self.vehicles.keys())
                assert given_keys == have_keys, "The input actions: {} have incompatible keys with existing {}!".format(
                    given_keys, have_keys
                )
            else:
                # That would be OK if extra actions is given. This is because, when evaluate a policy with naive
                # implementation, the "termination observation" will still be given in T=t-1. And at T=t, when you
                # collect action from policy(last_obs) without masking, then the action for "termination observation"
                # will still be computed. We just filter it out here.
                actions = {v_id: actions[v_id] for v_id in self.vehicles.keys()}
        return actions

    def _preprocess_macro_waypoints(self, waypoint_list: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray]]:
        if not self.is_multi_agent:
            # print('action.dtype: {}'.format(type(actions)))
            #print('action: {}'.format(actions))
            actions = waypoint_list
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        return actions

    def _step_macro_simulator(self, actions):
        #simulation_frequency = 30  # 60 80
        simulation_frequency = self.config['seq_traj_len']
        policy_frequency = 1
        frames = int(simulation_frequency / policy_frequency)
        self.time = 0
        # print('seq len is: ')
        # print(self.config['seq_traj_len'])
        #print('di action pairs: {}'.format(actions))
        #actions = {vid: self.action_type.actions[vvalue] for vid, vvalue in actions.items()}
        # wp_list = self.get_waypoint_list()
        # wps = dict()
        # for vid in actions.keys():
        #     wps[vid] = wp_list
        wps = actions
        # if isinstance(actions['default_agent'], dict):
        #     wps['default_agent'] = actions['default_agent']['raw_traj']
        for frame in range(frames):
            # we use frame to update robot position, and use wps to represent the whole trajectory
            scene_manager_before_step_infos = self.engine.before_step_macro(frame, wps)
            self.engine.step()
            scene_manager_after_step_infos = self.engine.after_step()
        #scene_manager_after_step_infos = self.engine.after_step()
        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

    def _get_reset_return(self):
        ret = {}
        self.engine.after_step()
        o = None
        o_reset = None
        print('episode reward: {}'.format(self.episode_rwd))
        self.episode_rwd = 0
        self.step_num = 0
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            ret[v_id] = self.observations[v_id].observe(v)
            o = self.observations[v_id].observe(v)
            self.update_current_state(v_id)
            self.vel_speed = 0
            if self.config["traj_control_mode"] == 'jerk':
                o_dict = {}
                o_dict['birdview'] = o 
                # v_state = np.zeros(4)
                # v_state[3] = v.last_spd
                # v_state = self.z_state
                v_state = np.zeros(6)
                # v_state[3] = 5.0
                o_dict['vehicle_state'] = v_state
                #o_dict['speed'] = v.last_spd
            elif self.config["traj_control_mode"] == 'acc':
                o_dict = {}
                o_dict['birdview'] = o 
                # v_state = np.zeros(4)
                # v_state[3] = v.last_spd
                # v_state = self.z_state[:6] #:4
                v_state = np.zeros(6)
                # v_state[3] = 5.0
                o_dict['vehicle_state'] = v_state
                #o_dict['speed'] = v.last_spd
            else:
                o_dict = o

            if self.config['use_explicit_vel_obs']:
                max_spd = 10.0
                cur_spd = self.z_state[3]
                a_pixel = cur_spd / max_spd
                a_pixel = clip(a_pixel, 0.0, 1.0)
                max_steer = 1.0
                cur_steer = self.z_state[5] + max_steer 
                b_pixel = cur_steer / max_steer 
                b_pixel = clip(b_pixel, 0.0, 1.0)
                additional_channel = np.zeros((200, 200, 1))
                original_channels = o_dict['birdview']
                combined_channel = np.concatenate((original_channels, additional_channel), axis = 2)
                o_dict['birdview'] = combined_channel
                o_dict['birdview'][:,:, 5]=0.0
                o_dict['birdview'][:,10:50, 5]=a_pixel
                o_dict['birdview'][:,50:90, 5]=b_pixel
                if self.config['use_explicit_vel_obs_compare']:
                    o_dict['birdview'][:,:, 5]=0.0

            o_reset = o_dict
            if hasattr(v, 'macro_succ'):
                v.macro_succ = False
            if hasattr(v, 'macro_crash'):
                v.macro_crash = False
            v.penultimate_state = {}
            v.penultimate_state['position'] = np.array([0,0])
            v.penultimate_state['yaw'] = 0 
            v.penultimate_state['speed'] = 0
            v.traj_wp_list = [] 
            v.traj_wp_list.append(copy.deepcopy(v.penultimate_state))
            v.traj_wp_list.append(copy.deepcopy(v.penultimate_state))
            v.last_spd = 0

        self.already_go_dist = 0
        self._compute_navi_dist = True 
        self.navi_distance = 100.0
        self.remove_init_stop = True
        self.episode_max_step = self.config['episode_max_step']
        self.z_xyt = np.zeros(3)
        if self.remove_init_stop:
            return o_reset
        return o_reset

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module, to avoid incompatible with ray
        if engine_initialized():
            return
        self.engine = initialize_engine(self.config)
        # engine setup
        self.setup_engine()
        # other optional initialization
        self._after_lazy_init()

    # def get_single_observation(self, _=None):
    #     o = TopDownMultiChannel(
    #         self.config["vehicle_config"],
    #         self.config["on_screen"],
    #         self.config["rgb_clip"],
    #         frame_stack=3,
    #         post_stack=10,
    #         frame_skip=1,
    #         resolution=(200, 200),
    #         max_distance=50
    #     )
    #     #o = TopDownMultiChannel(vehicle_config, self, False)
    #     return o
    
    def get_single_observation(self, _=None):
        o = HRLTopDownMultiChannel(
            self.config["vehicle_config"],
            self.config["on_screen"],
            self.config["rgb_clip"],
            frame_stack=3,
            post_stack=10,
            frame_skip=1,
            resolution=(200, 200),
            max_distance=50
        )
        return o

    def wrap_angle(self, angle_in_rad):
        #angle_in_rad = angle_in_degree / 180.0 * np.pi
        while (angle_in_rad > np.pi):
            angle_in_rad -= 2 * np.pi
        while (angle_in_rad <= -np.pi):
            angle_in_rad += 2 * np.pi
        return angle_in_rad

    def get_episode_max_step(self, distance, average_speed = 6.5):
        average_dist_per_step = float(self.config['seq_traj_len']) * average_speed * self.config['physics_world_step_size']
        max_step = int(distance / average_dist_per_step) + 1
        return max_step


register(
    id='HRL-v1',
    entry_point='core.envs.md_traj_env:MetaDriveTrajEnv',
)
