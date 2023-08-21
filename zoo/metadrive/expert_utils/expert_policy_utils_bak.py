from metadrive.policy.idm_policy import IDMPolicy, FrontBackObjects
import logging
import numpy as np
from metadrive.component.vehicle_module.PID_controller import PIDController

class ExpertIDMPolicy(IDMPolicy):

    def __init__(self, control_object, random_seed):
        super(ExpertIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 27 #15
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.LANE_CHANGE_FREQ = 100
        self.heading_pid = PIDController(1.2, 0.01, 2.5)
        self.lateral_pid = PIDController(0.2, .002, 0.05)
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
        
        # ego_vehicle = self.control_object
        # ego_pos = ego_vehicle.position
        # ego_theta = ego_vehicle.heading_theta 
        # v = ego_vehicle.last_spd
        # init_state = np.array([ego_pos[0], ego_pos[1], ego_theta, v])
        # final_state = self.plant_model(init_state, steering, acc)
        # ego_vehicle.last_spd = final_state[3]
        # ego_vehicle.set_position(np.array([final_state[0], final_state[1]]))
        # ego_vehicle.set_heading_theta(final_state[2])      
        # # ego_vehicle.speed = final_state[3] * 3.6
        # ego_vehicle.last_spd = final_state[3]
        # ego_vehicle.set_velocity([np.cos(final_state[2]), np.sin(final_state[2])], final_state[3]/3.6)
        # print('zt')
        
        return [steering, acc]
    
    def plant_model(self, init_state, steer, acc, dt = 0.1):
        x_t = init_state[0]
        y_t = init_state[1]
        psi_t = init_state[2]
        v_t = init_state[3]
        steering = steer * 0.5
        acceleration = acc * 5
        import numpy as np
        psi_dot = v_t * np.tan(steering) / 2.5 
        if psi_dot > 0.5:
            psi_dot = 0.5
        elif psi_dot < -0.5:
            psi_dot = -0.5
        psi_t_1 = psi_dot * 0.1 + psi_t 
        v_t_1 = v_t + acceleration * dt 
        x_dot = v_t_1 * np.cos(psi_t_1)
        y_dot = v_t_1 * np.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        if v_t_1 > 10.0:
            v_t_1 = 10.0
        elif v_t_1 < 0.0:
            v_t_1 = 0.0
        
        return np.array([x_t_1, y_t_1, psi_t_1, v_t_1])

    def set_target_speed(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        print('current speed: {}'.format(self.control_object.speed))
        print('normal speed: {}'.format(self.NORMAL_SPEED))
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
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST + 0
        else:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST - 0
