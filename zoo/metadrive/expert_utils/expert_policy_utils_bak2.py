from metadrive.policy.idm_policy import IDMPolicy, FrontBackObjects
import logging
import numpy as np
from metadrive.component.vehicle_module.PID_controller import PIDController
from zoo.metadrive.expert_utils.calculate_utils import calculate_fine_collision
from metadrive.utils.math_utils import not_zero, wrap_to_pi

class ExpertIDMPolicy(IDMPolicy):
    """
    We implement this policy based on the HighwayEnv code base.
    """
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.3  # [s]
    TAU_LATERAL = 0.8  # [s]
    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]
    DISTANCE_WANTED = 2.0 #10.0
    """Desired jam distance to the front vehicle."""
    TIME_WANTED =  0.5#1.5  # [s]
    """Desired time gap to the front v"""
    DELTA = 10.0  # []
    """Exponent of the velocity term."""
    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    LANE_CHANGE_FREQ = 50  # [step]
    LANE_CHANGE_SPEED_INCREASE = 2
    SAFE_LANE_CHANGE_DISTANCE = 10
    MAX_LONG_DIST = 30
    MAX_SPEED = 100
    # Normal speed
    NORMAL_SPEED = 30
    # Creep Speed
    CREEP_SPEED = 5
    # acc factor
    ACC_FACTOR = 1.0
    DEACC_FACTOR = -5

    def __init__(self, control_object, random_seed):
        super(ExpertIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 36#27 #15
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.LANE_CHANGE_FREQ = 5#100
        self.heading_pid = PIDController(1.2, 0.01, 2.5)
        self.lateral_pid = PIDController(0.2, .002, 0.05)
        self.use_const_ref = False
        self.need_reset_const_ref = False  
        self.const_ref = 0
        self.variation_in_line_type = None 
        self.misalign = False
        if self.control_object.navigation.expert_type == 'straight_agreesive':
            self.LANE_CHANGE_FREQ = 60
            self.NORMAL_SPEED_CONST = 27
            pass 
        elif self.control_object.navigation.expert_type == 'straight_wild':
            self.LANE_CHANGE_FREQ = 5000
            self.TIME_WANTED = 0.5
            self.DISTANCE_WANTED = 3.0
            self.use_const_ref = True
            self.need_reset_const_ref = True 
        elif self.control_object.navigation.expert_type == 'round_wild':
            self.LANE_CHANGE_FREQ = 5000
            self.TIME_WANTED = 0.5
            self.DISTANCE_WANTED = 3.0
            self.use_const_ref = True
            self.need_reset_const_ref = True    
            self.NORMAL_SPEED_CONST = 24     
        elif self.control_object.navigation.expert_type == 'round_agressive':
            self.NORMAL_SPEED_CONST = 30 
            self.variation_in_line_type = False  
        elif self.control_object.navigation.expert_type == 'inter_wild':
            self.use_const_ref = True
            self.need_reset_const_ref = True      
            self.NORMAL_SPEED_CONST = 26    
            self.enable_lane_change = False      
        elif self.control_object.navigation.expert_type == 'inter_agressive':    
            self.NORMAL_SPEED_CONST = 30                   
            # current_lanes = self.control_object.navigation.current_ref_lanes
            # import random
            # self.label_ref_lane_num = random.randint(0, len(current_lanes))
            

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
        
        objs_interest = self.check_in_fan(all_objects)
        collision = False
        for other_vehicle in objs_interest:
            if(self.check_single_collision(other_vehicle)):
                collision = True 
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        if collision:
            acc = -1.0
        if steering > 0.5:
            steering = 0.5
            # print('zt steering > 0.5')
        if steering < -0.5:
            steering = -0.5
            # print('zt steering < -0.5')
        return [steering, acc]

    def move_to_next_road(self):
        # routing target lane is in current ref lanes
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            if self.use_const_ref:
                if self.need_reset_const_ref is True:
                    import random 
                    random_number = random.randint(0, len(current_lanes)-1)
                    self.const_ref = random_number
                    self.need_reset_const_ref = False 
                self.routing_target_lane = current_lanes[self.const_ref]
            return True if self.routing_target_lane in current_lanes else False
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane):
                    # two lanes connect
                    self.routing_target_lane = lane
                    return True
                    # lane change for lane num change
            return False
        elif self.control_object.lane in current_lanes and self.routing_target_lane is not self.control_object.lane:
            # lateral routing lane change
            self.routing_target_lane = self.control_object.lane
            if self.use_const_ref:
                self.routing_target_lane = current_lanes[self.const_ref]
            self.overtake_timer = self.np_random.randint(0, int(self.LANE_CHANGE_FREQ / 2))
            return True
        else:
            return True

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.ACC_FACTOR * self.DEACC_FACTOR
        dv = np.dot(ego_vehicle.velocity - front_obj.velocity, ego_vehicle.heading) if projected \
            else ego_vehicle.speed - front_obj.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def reset(self):
        self.heading_pid.reset()
        self.lateral_pid.reset()
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        if self.use_const_ref is True:
            self.need_reset_const_ref = True 
        if self.variation_in_line_type is not None:
            self.variation_in_line_type = False

    # def lane_change_policy(self, all_objects):
    #     current_lanes = self.control_object.navigation.current_ref_lanes
    #     surrounding_objects = FrontBackObjects.get_find_front_back_objs(
    #         all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
    #     )
    #     self.available_routing_index_range = [i for i in range(len(current_lanes))]
    #     next_lanes = self.control_object.navigation.next_ref_lanes
    #     lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0

    #     # We have to perform lane changing because the number of lanes in next road is less than current road
    #     if lane_num_diff > 0:
    #         # lane num decreasing happened in left road or right road
    #         if current_lanes[0].is_previous_lane_of(next_lanes[0]):
    #             index_range = [i for i in range(len(next_lanes))]
    #         else:
    #             index_range = [i for i in range(lane_num_diff, len(current_lanes))]
    #         self.available_routing_index_range = index_range
    #         if self.routing_target_lane.index[-1] not in index_range:
    #             # not on suitable lane do lane change !!!
    #             if self.routing_target_lane.index[-1] > index_range[-1]:
    #                 # change to left
    #                 if surrounding_objects.left_back_min_distance(
    #                 ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
    #                     # creep to wait
    #                     self.target_speed = self.CREEP_SPEED
    #                     return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
    #                     ), self.routing_target_lane
    #                 else:
    #                     # it is time to change lane!
    #                     self.target_speed = self.NORMAL_SPEED
    #                     return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
    #                            current_lanes[self.routing_target_lane.index[-1] - 1]
    #             else:
    #                 # change to right
    #                 if surrounding_objects.right_back_min_distance(
    #                 ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
    #                     # unsafe, creep and wait
    #                     self.target_speed = self.CREEP_SPEED
    #                     return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
    #                     ), self.routing_target_lane,
    #                 else:
    #                     # change lane
    #                     self.target_speed = self.NORMAL_SPEED
    #                     return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
    #                            current_lanes[self.routing_target_lane.index[-1] + 1]

    #     # lane follow or active change lane/overtake for high driving speed
    #     if abs(self.control_object.speed - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
    #     ) and abs(surrounding_objects.front_object().speed -
    #               self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
    #         # may lane change
    #         right_front_speed = surrounding_objects.right_front_object().speed if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
    #             if surrounding_objects.right_lane_exist() and surrounding_objects.right_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
    #         front_speed = surrounding_objects.front_object().speed if surrounding_objects.has_front_object(
    #         ) else self.MAX_SPEED
    #         left_front_speed = surrounding_objects.left_front_object().speed if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
    #             if surrounding_objects.left_lane_exist() and surrounding_objects.left_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
    #         if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
    #             # left overtake has a high priority
    #             expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
    #             if expect_lane_idx in self.available_routing_index_range:
    #                 return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
    #                        current_lanes[expect_lane_idx]
    #         if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
    #             expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
    #             if expect_lane_idx in self.available_routing_index_range:
    #                 return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
    #                        current_lanes[expect_lane_idx]

    #     # fall back to lane follow
    #     self.target_speed = self.NORMAL_SPEED
    #     self.overtake_timer += 1
    #     return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane

    def lane_change_policy(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0
        
        if self.variation_in_line_type is not None:
            variate_type = False
            zt_lane = self.control_object.lane
            current_long = zt_lane.local_coordinates(self.control_object.position)[0]
            lane_heading = zt_lane.heading_theta_at(current_long)
            
            
            
            v_heading = self.control_object.heading_theta
            heading_err = np.abs(lane_heading - v_heading)
            if self.control_object.lane.index[1] in ['1O0_0_', '2S0_0_']:
                self.misalign = True 
            else:
                self.misalign = False
            # self.misalign = True if heading_err > 0.2 else False
            # self.misalign = False
            if next_lanes is None:
                variate_type = False
            elif type(current_lanes[0]).__name__ != type(next_lanes[0]).__name__ :
                if zt_lane.length - current_long < 20:
                    variate_type = True
            self.variation_in_line_type = variate_type

        # We have to perform lane changing because the number of lanes in next road is less than current road
        if lane_num_diff > 0:
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane
                    else:
                        # it is time to change lane!
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1]
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane,
                    else:
                        # change lane
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1]

        # lane follow or active change lane/overtake for high driving speed
        # if abs(self.control_object.speed - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
        # ) and abs(surrounding_objects.front_object().speed -
        #           self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
        if abs(self.control_object.speed - self.NORMAL_SPEED) > 0 and surrounding_objects.has_front_object(
        ) and abs(surrounding_objects.front_object().speed -
                  self.NORMAL_SPEED) > 0 and self.overtake_timer > self.LANE_CHANGE_FREQ:
            # may lane change
            right_front_speed = surrounding_objects.right_front_object().speed if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_lane_exist() and surrounding_objects.right_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
            front_speed = surrounding_objects.front_object().speed if surrounding_objects.has_front_object(
            ) else self.MAX_SPEED
            left_front_speed = surrounding_objects.left_front_object().speed if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_lane_exist() and surrounding_objects.left_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
            if left_front_speed is not None and surrounding_objects.left_front_min_distance() - surrounding_objects.front_min_distance() >0 and surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE - 3:
                # left overtake has a high priority
                expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                # self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
                if expect_lane_idx in self.available_routing_index_range:
                    return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                           current_lanes[expect_lane_idx]
            if right_front_speed is not None and surrounding_objects.right_front_min_distance() - surrounding_objects.front_min_distance() >0 and surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE - 3:
                expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                # self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
                if expect_lane_idx in self.available_routing_index_range:
                    return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                           current_lanes[expect_lane_idx]

        # fall back to lane follow
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane

    def set_target_speed(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        # print('current speed: {}'.format(self.control_object.speed))
        # print('normal speed: {}'.format(self.NORMAL_SPEED))
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )    
        current_lane = self.control_object.lane
        total_lane_num = len(current_lanes)
        current_lane_idx = current_lane.index[-1]
        if current_lane_idx == 0 or current_lane_idx == current_lane_idx-1:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        elif current_lane_idx % 2 == 0:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        else:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        if self.variation_in_line_type is not None:
            if self.variation_in_line_type is True:
                self.NORMAL_SPEED = 20
                print('variate in line type')

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        #lane_heading = target_lane.heading_theta_at(long + 1)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading)) * 1.5
        steering += self.lateral_pid.get_result(-lat) * 1.0 #0.6
        #print('zt')
        return float(steering)


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
        for t in [1.0, 2,0, 3.0, 4.0]:
            e_fp = self.get_ego_future_position(t)
            o_fp = self.get_future_position(other_vehicle, t)
            distance = (e_fp[0] - o_fp[0]) ** 2 + (e_fp[1] - o_fp[1]) ** 2
            collision_thred = 7.0 * 7.0
            if distance < collision_thred:
                #collision = True
                collision_label = calculate_fine_collision(self.control_object, other_vehicle, e_fp, o_fp)
                if collision_label:
                    if self.control_object.navigation.expert_type == 'round_wild':
                        collision = True
                    if self.control_object.navigation.expert_type == 'round_agressive':
                        if self.variation_in_line_type is not None:
                            if self.variation_in_line_type is True:
                                collision = True
                        if self.misalign:
                            collision = True
                    if self.control_object.navigation.expert_type == 'inter_wild':
                        collision = True
                    if self.control_object.navigation.expert_type == 'inter_agressive':
                        collision = True
        return collision 