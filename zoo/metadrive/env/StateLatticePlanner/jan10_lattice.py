"""

State lattice planner with model predictive trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

- plookuptable.csv is generated with this script:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning
/ModelPredictiveTrajectoryGenerator/lookup_table_generator.py

Ref:

- State Space Sampling of Feasible Motions for High-Performance Mobile Robot
Navigation in Complex Environments
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.187.8210&rep=rep1
&type=pdf

"""
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import ModelPredictiveTrajectoryGenerator.trajectory_generator as planner
import ModelPredictiveTrajectoryGenerator.motion_model as motion_model

TABLE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/lookup_table.csv"

show_animation = True


pwd = os.getcwd()
DATA_PATH = pwd + '/data_folder/'
PATH_LIBRARY_PATH = pwd + '/dataset/raw_data/path_matfiles/'
print(PATH_LIBRARY_PATH)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(PATH_LIBRARY_PATH):
    os.makedirs(PATH_LIBRARY_PATH)
mode = 'draw' 
# mode = 'save'


def search_nearest_one_from_lookup_table(t_x, t_y, t_yaw, lookup_table):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(lookup_table):
        dx = t_x - table[0]
        dy = t_y - table[1]
        dyaw = t_yaw - table[2]
        d = math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2)
        if d <= mind:
            minid = i
            mind = d

    return lookup_table[minid]


def get_lookup_table(table_path):
    return np.loadtxt(table_path, delimiter=',', skiprows=1)


def generate_path(target_states, k0):
    # x, y, yaw, s, km, kf
    lookup_table = get_lookup_table(TABLE_PATH)
    result = []

    for state in target_states:
        bestp = search_nearest_one_from_lookup_table(
            state[0], state[1], state[2], lookup_table)

        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        init_p = np.array(
            [np.hypot(state[0], state[1]), bestp[4], bestp[5]]).reshape(3, 1)

        x, y, yaw, p = planner.optimize_trajectory(target, k0, init_p)

        if x is not None:
            print("find good path")
            result.append(
                [x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])

    print("finish path generation")
    return result


def calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy):
    """

    calc lane states

    :param l_center: lane lateral position
    :param l_heading:  lane heading
    :param l_width:  lane width
    :param v_width: vehicle width
    :param d: longitudinal position
    :param nxy: sampling number
    :return: state list
    """
    xc = d
    yc = l_center

    states = []
    for i in range(nxy):
        delta = -0.5 * (l_width - v_width) + \
            (l_width - v_width) * i / (nxy - 1)
        xf = xc - delta * math.sin(l_heading)
        yf = yc + delta * math.cos(l_heading)
        yawf = l_heading
        states.append([xf, yf, yawf])

    return states



def lane_state_sampling_test4():
    k0 = 0.0

    l_center = 2.4
    l_heading = np.deg2rad(60.0)
    l_width = 5.0
    v_width = 1.0
    d = 5
    nxy = 7
    states = calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy)
    result = generate_path(states, k0)

    if show_animation:
        plt.close("all")

    for table in result:
        x_c, y_c, yaw_c = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)
        
        x_new_c = []
        y_new_c = []
        # x_new_c.append(x_c[0])
        # x_new_c.append(x_c[1])
        # x_new_c.append(x_c[2])
        # x_new_c.append(x_c[3])
        # x_new_c.append(x_c[4])
        # x_new_c.append(x_c[5])
        for i in range(8, len(x_c)):
            x_new_c.append(x_c[i-8] * 0.6)

        # y_new_c.append(y_c[0])
        # y_new_c.append(y_c[7])
        # y_new_c.append(y_c[10])
        # y_new_c.append(y_c[12])
        # y_new_c.append(y_c[14])
        # y_new_c.append(y_c[15])
        for i in range(8, len(y_c)):
            y_new_c.append((y_c[i]-y_c[8])*0.6)



        # if show_animation:
        #     plt.plot(x_c, y_c, "-r")
        if show_animation:
            plt.plot(x_new_c, y_new_c, "-r")

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()

def lane_state_sampling_one_case(final_theta_degree, dd =20):
    print('degree:')
    print(final_theta_degree)
    k0 = 0.0
    l_center = 0.1 * final_theta_degree #2.4
    l_center = 0.04 * final_theta_degree #2.4
    # l_center = 0.04 * final_theta_degree #2.4
    l_heading = np.deg2rad(final_theta_degree)
    l_width = 10.0#5.0
    v_width = 1.0
    d = dd
    # d = 20
    # d =5
    # d = 6
    nxy = 25
    states = calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy)
    result = generate_path(states, k0)

    if show_animation:
        plt.close("all")
    path_list = []

    for table in result:
        x_c, y_c, yaw_c = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)
        
        x_new_c = []
        y_new_c = []
        theta_new_c = []
        for i in range(8, len(x_c)):
            x_new_c.append(x_c[i-8] * 1.0)
        for i in range(8, len(y_c)):
            y_new_c.append((y_c[i]-y_c[8])*1.0)
            theta_new_c.append(yaw_c[i])

        for i in range(50):
            x_new_c.append(x_new_c[-1] + 0.1)
            y_new_c.append(y_new_c[-1] + 0.1 * np.tan(theta_new_c[-1]))
            theta_new_c.append(theta_new_c[-1])
        
        x_new_c = np.array(x_new_c)
        y_new_c = np.array(y_new_c)
        theta_new_c = np.array(theta_new_c)

        lon = np.expand_dims(x_new_c,1)
        lat = np.expand_dims(y_new_c,1)
        yaw = np.expand_dims(theta_new_c,1)
        yaw = yaw * 180 / 3.1415926
        path = np.hstack((lon, lat, yaw))
        path_list.append(path)






        # if show_animation:
        #     plt.plot(x_c, y_c, "-r")
        if show_animation:
            plt.plot(x_new_c, y_new_c, "-r")
    # return path_list

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    return path_list



def main():
    import copy
    import scipy.io as io
    planner.show_animation = show_animation
    degree_list = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    #degree_list = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
    dd = 12 #20， 16， 12
    for degree in degree_list:
        path_id = 0
        print('preparing degree {}'.format(degree))
        path_dictionary = {}
        path_list = lane_state_sampling_one_case(degree, dd)
        for path in path_list:
            if path is not None:
                path_dictionary[str(path_id)] = copy.deepcopy(path)
                path_id += 1 
        mat_name = PATH_LIBRARY_PATH +"le8n" + str(dd)+"degree" + str(degree) + "mat.mat"   
        io.savemat(mat_name, path_dictionary) 



if __name__ == '__main__':
    main()
