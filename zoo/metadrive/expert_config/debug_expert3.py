"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import random

import numpy as np

from metadrive import MetaDriveEnv
from zoo.metadrive.env.expert_traj_env import MetaDriveTrajEnv
from metadrive.constants import HELP_MESSAGE
from zoo.metadrive.utils.traffic_manager_utils import TrafficMode

expert_data_folder = "/home/hunter/jan_feb/temp_ckpt/xad_expert_data_expcc"
if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=True,
        traffic_density=0.40, #0.55,
        environment_num=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        expert_type = 'straight_agreesive',  
        avg_speed = 3.0,
        map='SSSSSSSSSSSSS',
        #expert_type = 'straight_wild',
        #expert_type = 'round_wild',
        #expert_type = 'round_agressive',
        # map=4,  # seven block
        #map='OSOS',
        # map='SXSX',
        # enable_u_turn = True,
        # traffic_mode = TrafficMode.Trigger,
        start_seed=random.randint(0, 1000),
        save_expert_data = True, 
        expert_data_folder = expert_data_folder,
        seq_traj_len=20,
        show_seq_traj = True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(offscreen_render=True))
    env = MetaDriveTrajEnv(config)
    # try:
    o = env.reset()
    print(HELP_MESSAGE)
    env.vehicle.expert_takeover = True
    # if args.observation == "rgb_camera":
    #     assert isinstance(o, dict)
    #     print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
    # else:
    #     assert isinstance(o, np.ndarray)
    #     print("The observation is an numpy array with shape: ", o.shape)
    for i in range(1, 1000000000):
        o, r, d, info = env.step([0, 0])
        # if d and info["arrive_dest"]:
        if d:
            env.reset()
            env.current_track_vehicle.expert_takeover = True
    # except:
    #     pass
    # finally:
    #     env.close()
