import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib import cm
import scipy.io as io

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

def generate_vel():
    zt_scale = 7.0
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

def select_trajectory_from_path(path_ele):
    vel_ele = generate_vel()
    return combine_trajectory(path_ele, vel_ele)

