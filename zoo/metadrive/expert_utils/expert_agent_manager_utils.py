from metadrive.manager.agent_manager import AgentManager
from zoo.metadrive.utils.macro_policy import ManualMacroDiscretePolicy
from zoo.metadrive.utils.vehicle_utils import MacroDefaultVehicle
from metadrive.utils.space import ParameterSpace, VehicleParameterSpace
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.utils import Config, safe_clip_for_small_array
from typing import Union, Dict, AnyStr, Tuple
from zoo.metadrive.expert_utils.expert_policy_utils import ExpertIDMPolicy

class ExpertAgentManager(AgentManager):
    def _get_policy(self, obj):
        policy = ExpertIDMPolicy(obj, self.generate_seed())
        return policy
    def before_step(self, frame = 0, wps=None):
        # not in replay mode
        self._agents_finished_this_frame = dict()
        step_infos = {}
        for agent_id in self.active_agents.keys():
            policy = self.engine.get_policy(self._agent_to_object[agent_id])
            # action = policy.act(agent_id)
            action = policy.act()
            step_infos[agent_id] = policy.get_action_info()
            step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))
        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1
            if self._dying_objects[v_name][1] == 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self._dying_objects[v_name][0]
                self._remove_vehicle(v)
                finished.add(v_name)
        for v_name in finished:
            self._dying_objects.pop(v_name)
        return step_infos

    def _get_vehicles(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        v_type = MacroDefaultVehicle
        for agent_id, v_config in config_dict.items():
            obj = self.spawn_object(v_type, vehicle_config=v_config)
            ret[agent_id] = obj
            policy = self._get_policy(obj)
            self.engine.add_policy(obj.id, policy)
        return ret
