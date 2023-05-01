from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
import os
import sys
from typing import Tuple

import gym
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import Decoration, DEFAULT_AGENT, EDITION
from metadrive.obs.observation_base import ObservationBase
from metadrive.obs.top_down_obs_impl import WorldSurface, ObservationWindow, COLOR_BLACK, \
    VehicleGraphics, LaneGraphics
from metadrive.utils import import_pygame

pygame = import_pygame()


class HRLTopDownMultiChannel(TopDownMultiChannel):

    def render(self) -> np.ndarray:
        if self.onscreen:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()

        if hasattr(self.target_vehicle.navigation, 'should_redraw'):
            if self.target_vehicle.navigation.should_redraw is True:
                self._should_draw_map = True

        if self._should_draw_map:
            self.draw_map()

        if hasattr(self.target_vehicle.navigation, 'should_redraw'):
            #if self.target_vehicle.navigation.should_redraw is True:
            self.target_vehicle.navigation.should_redraw = False

        self.draw_scene()

        if self.onscreen:
            self.screen.fill(COLOR_BLACK)
            screen = self.obs_window.get_screen_window()
            if screen.get_size() == self.screen.get_size():
                self.screen.blit(screen, (0, 0))
            else:
                pygame.transform.scale2x(self.obs_window.get_screen_window(), self.screen)
            pygame.display.flip()


    def draw_navigation(self, canvas, color=(128, 128, 128)):
        checkpoints = self.target_vehicle.navigation.checkpoints
        if hasattr(self.target_vehicle.navigation, 'should_redraw'):
            if self.target_vehicle.navigation.should_redraw is True:
                if hasattr(self.target_vehicle.navigation, 'temp_checkpoints'):
                    checkpoints = self.target_vehicle.navigation.temp_checkpoints
        for i, c in enumerate(checkpoints[:-1]):
            lanes = self.road_network.graph[c][checkpoints[i + 1]]
            for lane in lanes:
                LaneGraphics.simple_draw(lane, canvas, color=color)