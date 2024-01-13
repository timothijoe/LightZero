import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib import cm
import scipy.io as io
import copy
from metadrive.policy.idm_policy import FrontBackObjects


is_left_first = True 

###########################   Coordinate Transformation   ################################################################
##########################################################################################################################
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
        wp_w = convert_wp_to_world_coord(wp, rbt_pos)
        wp_w_list.append(wp_w)
    return wp_w_list

def odom_to_local(point2d, robot_pos):
    local_point = [0.0, 0.0]
    yaw_robot = robot_pos[2]
    delta_x = point2d[0] - robot_pos[0]
    delta_y = point2d[1] - robot_pos[1]
    #delta_yaw = goal[2] - robot_pos[2]
    #local_goal[2] = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
    local_point[0] = delta_x * math.cos(yaw_robot) + delta_y * math.sin(yaw_robot)
    local_point[1] = -delta_x * math.sin(yaw_robot) + delta_y * math.cos(yaw_robot)
    return local_point
##########################################################################################################################




###########################   Path, Vel, Trajectory generation  ##########################################################
##########################################################################################################################
class MetaPath():
    def __init__(self, wp_list):
        self.path = wp_list 
        self.path_length = self.getLen()
        self.total_path_len = self.path_length[-1]

    def GetPosFromLength(self, s):
        cal_length = lambda pos:(np.sum(np.sqrt(np.sum(np.square((pos[1:,:2] - pos[:-1,:2])),axis=1)),axis=0))
        if(self.path_length[-1] - s < 0):
            print(self.path_length[-1])
            print('zt')
            print(s)
        matched_ind = min(np.where(np.array(self.path_length)-s >= 0)[0])
        if matched_ind == 0:
            return self.path[:1,:]
        else:
            pos = np.zeros((1,3))         
            percent = (s - cal_length(self.path[:matched_ind, :])) / cal_length(self.path[matched_ind-1:matched_ind+1, :])
            pos[0,0] = self.path[matched_ind-1, 0] + percent * (self.path[matched_ind, 0] - self.path[matched_ind-1, 0])
            pos[0,1] = self.path[matched_ind-1, 1] + percent * (self.path[matched_ind, 1] - self.path[matched_ind-1, 1])
            pos[0,2] = self.path[matched_ind-1, 2] + percent * (self.path[matched_ind, 2] - self.path[matched_ind-1, 2])
            return pos
    def getLen(self):
        cal_length = lambda pos:(np.sum(np.sqrt(np.sum(np.square((pos[1:,:2] - pos[:-1,:2])),axis=1)),axis=0))
        path_length = [cal_length(self.path[:i+1,:2]) for i in range(len(self.path-1))]
        return path_length            
    def PlotPath(self):
        plt.title('Path Profile')
        plt.plot(self.path[:,0], self.path[:,1])

class MetaVel():
    def __init__(self, vel_list):
        self.vel_list = vel_list 
        self.t = self.vel_list[0,:]
        self.s = self.vel_list[1,:]
        self.milestone = [0]
        self.total_dist = self.getIntegral()
        self.T = self.t[-1]

    def getIntegral(self):
        self.dt = self.t[1] - self.t[0]
        dist = 0
        for speed in self.s:
            dist += speed * self.dt
            self.milestone.append(dist)
        return dist

    def get_speed_list(self):
        return self.s
    def GetDistance(self, t):
        if t > self.T:
            raise Exception ('GetSpeed method: The specified time exceeds the time horizon of speed profile')
        else:
            return self.milestone[int(round(t/self.dt))]
    def GetSpeed(self, t):
        if t > self.T:
            raise Exception ('GetSpeed method: The specified time exceeds the time horizon of speed profile')
        else:
            return self.s[int(round(t/self.dt))]
    def PlotSpeed(self):
        plt.title('Speed Profile')
        plt.plot(self.t, self.s)

def generate_vel(zt_scale = 7.0, acc = 0):
    # zt_scale = 7.0
    if acc > 2.0:
        acc = 2.0
    elif acc < -2.0:
        acc = -2.0
    total_time = 2.0
    step = 0.1
    times = [i for i in np.arange(0, total_time + step, step)]
    initial_speed = zt_scale 
    speeds = [max(min(initial_speed + acc * t, 8),0) for t in times]

    arr1 = np.array(times)
    arr2 = np.array(speeds)


    # arr1 = np.arange(0, 2.1, 0.1)
    # arr2 = np.ones(len(arr1), dtype=float) * zt_scale
    result = np.vstack((arr1, arr2))
    return result 

def generate_vel_v2(zt_scale = 7.0):
    # zt_scale = 7.0
    arr1 = np.arange(0, 2.1, 0.1)
    arr2 = np.ones(len(arr1), dtype=float) * zt_scale
    result = np.vstack((arr1, arr2))
    return result 

def combine_trajectory(path_ele, vel_ele):
    path = MetaPath(path_ele)
    vel = MetaVel(vel_ele)
    Ros = np.zeros((1,5))
    SpeedCoveredDist = []
    Ros[0][3] = vel.s[0]
    for t in vel.t:
        if t == 0.0:
            continue
        SpeedCoveredDist.append(vel.GetDistance(t))
        cur_vel = vel.GetSpeed(t)
        cur_vel = np.array([[cur_vel]])
        cur_time = np.array([[t]])
        pose_twist = np.hstack((path.GetPosFromLength(SpeedCoveredDist[-1]),cur_vel, cur_time))
        Ros = np.vstack((Ros, pose_twist))
    return Ros[:, :2]

def select_trajectory_from_path(path_ele, init_speed = 7.0, acc = 0):
    vel_ele = generate_vel(init_speed, acc)
    return combine_trajectory(path_ele, vel_ele)


###########################   Lane information and lane chone   ##########################################################
##########################################################################################################################


def get_lane_lateral_pos(vehicle, robot_pos, path_dict):
    # naive justification, in straight highway
    degree = -robot_pos[2] * 180 / 3.1415926
    ego_speed_ms = robot_pos[3]
    rbt_pos = copy.deepcopy(robot_pos)
    forcast = 15 
    current_lane = vehicle.lane
    vehicle_status, vehicle_near_speed, vehicle_near_distance = justify_if_lanes_ok(vehicle)
    current_lanes = vehicle.navigation.current_ref_lanes 
    total_lane_num = len(current_lanes)
    current_lane = vehicle.lane 
    current_lane_idx = current_lane.index[-1]
    valid_idx = 0
    # if vehicle_status[0]:
    #     valid_idx = 0
    # elif vehicle_status[2]:
    #     valid_idx = 2
    # else:
    #     valid_idx = 1
    global is_left_first
    if is_left_first:
        if vehicle_status[0]:
            valid_idx = 0
        elif vehicle_status[2]:
            valid_idx = 2
        else:
            valid_idx = 1
    else:
        if vehicle_status[2]:
            valid_idx = 2
        elif vehicle_status[0]:
            valid_idx = 0
        else:
            valid_idx = 1      
    ref_current_lane_idx = valid_idx + current_lane_idx -1

    y_list = []
    long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
    for index in range(len(vehicle.navigation.current_ref_lanes)):
        if not index == ref_current_lane_idx:
            continue
        long_target = long_now + forcast
        lateral_target = 0 
        position_target = vehicle.navigation.current_ref_lanes[index].position(long_target, lateral_target) 
        position_local = odom_to_local(position_target, rbt_pos)
        y_target = position_local[1]
        y_list.append(y_target)

    min_distance = float('inf')
    yaw_key = None 
    for key in path_dict.keys():
        key_float = float(key)
        distance = np.abs(key_float - degree)
        if distance < min_distance:
            min_distance = distance 
            yaw_key = key      
    key_y_list = []
    for y in y_list:
        min_distance = float('inf')
        y_key = None                    
        for key in path_dict[yaw_key].keys():
            key_float = float(key)
            distance = np.abs(key_float - y)
            if distance < min_distance:
                min_distance = distance 
                y_key = key 
        key_y_list.append(copy.deepcopy(y_key))    
    selected_keys = key_y_list 
    zt_traj_list = []  

    acc = calculate_acc_for_maintaing_distance(vehicle_near_distance[valid_idx], ego_speed_ms, vehicle_near_speed[valid_idx]/3.6, desired_distance = 12, t=2)
    
    for key in selected_keys:
        zt_path = path_dict[yaw_key][key]['path']
        zt_traj =  select_trajectory_from_path(zt_path, ego_speed_ms, acc)
        zt_traj_list.append(zt_traj)
    return zt_traj_list 

def justify_if_lanes_ok(vehicle):
    vehicle_status = [False, False, False]
    vehicle_nearby_speeds = [None, None, None]
    vehicle_nearby_distance = [None, None, None]
    all_objects=vehicle.lidar.get_surrounding_objects(vehicle) 
    current_lane = vehicle.lane
    surrounding_objects = FrontBackObjects.get_find_front_back_objs(
        all_objects,
        current_lane,# lane
        vehicle.position, # position
        30, # max_distance
        vehicle.navigation.current_ref_lanes,
    )
    SAFE_LANE_CHANGE_DISTANCE = 15
    if surrounding_objects.right_lane_exist() and surrounding_objects.right_front_min_distance() > SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.right_back_min_distance() > SAFE_LANE_CHANGE_DISTANCE:
        # print("can turn right")
        vehicle_status[2] = True
        vehicle_nearby_speeds[2] = surrounding_objects.right_front_object().speed if surrounding_objects.right_front_object() else 8.0
        vehicle_nearby_distance[2] = surrounding_objects.right_front_min_distance()
    else:
        # print("can't turn right !!!")
        vehicle_status[2] = False
        vehicle_nearby_speeds[2] = None
        
    if surrounding_objects.left_lane_exist() and surrounding_objects.left_front_min_distance() > SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.left_back_min_distance() > SAFE_LANE_CHANGE_DISTANCE:    
        # print("can turn left")
        vehicle_status[0] = True
        vehicle_nearby_speeds[0] = surrounding_objects.left_front_object().speed if surrounding_objects.left_front_object() else 8.0
        vehicle_nearby_distance[0] = surrounding_objects.left_front_min_distance()
    else:
        vehicle_status[0] = False
        vehicle_nearby_speeds[0] = None 

    if surrounding_objects.front_min_distance() > SAFE_LANE_CHANGE_DISTANCE - 5:
        vehicle_status[1] = True 
        vehicle_nearby_speeds[1] = surrounding_objects.front_object().speed if surrounding_objects.front_object() else 8.0
        vehicle_nearby_distance[1] = surrounding_objects.front_min_distance()
    else:
        vehicle_status[1] = False
        vehicle_nearby_speeds[1] = surrounding_objects.front_object().speed if surrounding_objects.front_object() else 8.0
        vehicle_nearby_distance[1] = surrounding_objects.front_min_distance()
    
    global is_left_first
    if not surrounding_objects.left_lane_exist() and is_left_first:
        is_left_first = False
    if not surrounding_objects.right_lane_exist() and not is_left_first:
        is_left_first = True

    return vehicle_status, vehicle_nearby_speeds, vehicle_nearby_distance

def calculate_acc_for_maintaing_distance(front_distance, v_ego, v_other, desired_distance = 5, t=2):
    s_other = v_other * t 
    final_position_front = front_distance + s_other 
    final_position_ego = final_position_front - desired_distance 
    s1_needed = final_position_ego 
    a = (s1_needed - v_ego * t) * 2 / (t**2)
    return a 