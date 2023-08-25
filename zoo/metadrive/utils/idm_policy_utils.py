from metadrive.policy.idm_policy import IDMPolicy, FrontBackObjects
import logging
import numpy as np
from zoo.metadrive.expert_utils.calculate_utils import calculate_fine_collision

class MacroIDMPolicy(IDMPolicy):

    def __init__(self, control_object, random_seed):
        super(MacroIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 18 #15
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.LANE_CHANGE_FREQ = 300

    def act(self, *args, **kwargs):
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                self.set_target_speed(all_objects)
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            logging.warning("IDM bug! fall back")
            print("IDM bug! fall back")

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        
        # objs_interest = self.check_in_fan(all_objects)
        # collision = False
        # for other_vehicle in objs_interest:
        #     if(self.check_single_collision(other_vehicle)):
        #         collision = True 
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        # if collision:
        #     acc = -1.0        
        
        return [steering, acc]

    def set_target_speed(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )
        # if not surrounding_objects.right_lane_exist():
        #     self.NORMAL_SPEED = self.NORMAL_SPEED_CONST -4.0
        # elif not surrounding_objects.left_lane_exist():
        #     self.NORMAL_SPEED = self.NORMAL_SPEED_CONST + 4.0
        # else:
        #     self.NORMAL_SPEED = self.NORMAL_SPEED_CONST + 4 * (2 * np.random.rand() - 1)
            
        current_lane = self.control_object.lane
        total_lane_num = len(current_lanes)
        current_lane_idx = current_lane.index[-1]
        if current_lane_idx == 0 or current_lane_idx == current_lane_idx-1:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        elif current_lane_idx % 2 == 0:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST + 2
        else:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST - 2

    def check_in_fan(self, all_objects):
        valid_obj = []
        for other_vehicle in all_objects:
            e_fp = self.get_ego_future_position(0.0)
            o_fp = self.get_future_position(other_vehicle, 0.0)
            diff = o_fp - e_fp 
            vx_self = self.control_object.velocity[0] / 3.6
            vy_self = self.control_object.velocity[1] / 3.6 
            valid = False
            if diff[0] * vx_self + diff[1] * vy_self > 0:
                valid = True 
            if valid and np.linalg.norm(diff) < 35:
                valid_obj.append(other_vehicle)
        return valid_obj 

    def get_future_position(self, other_vehicle, dt = 4.0):
        x_other = other_vehicle.position[0]
        y_other = other_vehicle.position[1]
        vx_other = other_vehicle.velocity[0] / 3.6
        vy_other = other_vehicle.velocity[1] / 3.6
        x_future = x_other + vx_other * dt
        y_future = y_other + vy_other * dt
        future_pos = np.array([x_future, y_future])
        return future_pos

    def get_ego_future_position(self, dt = 4.0):
        x_self = self.control_object.position[0]
        y_self = self.control_object.position[1]
        vx_self = self.control_object.velocity[0] / 3.6
        vy_self = self.control_object.velocity[1] / 3.6
        x_future = x_self + vx_self * dt
        y_future = y_self + vy_self * dt
        future_pos = np.array([x_future, y_future])
        return future_pos

    def check_single_collision(self, other_vehicle):
        collision = False
        for t in [1.0]:
            e_fp = self.get_ego_future_position(t)
            o_fp = self.get_future_position(other_vehicle, t)
            distance = (e_fp[0] - o_fp[0]) ** 2 + (e_fp[1] - o_fp[1]) ** 2
            collision_thred = 7.0 * 7.0
            if distance < collision_thred:
                #collision = True
                collision_label = calculate_fine_collision(self.control_object, other_vehicle, e_fp, o_fp)
                # if collision_label:
                #     collision = True
        return collision 