#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import time
import threading
from enum import Enum

from ..utils import TeleopEvents
from .joncon_utils import JoyconRobotics


def math_clip(value, min_value, max_value):
    """Clamp a value between min_value and max_value."""
    return max(min(value, max_value), min_value)


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def should_quit(self):
        """Return True if the user has requested to quit."""
        return not self.running

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        else:
            return "close"


class JoyconController(InputController):
    """Joycon controller that wraps JoyconRobotics for basic 3DOF control."""

    def __init__(
        self,
        device="right",
        gripper_open=1.0,
        gripper_close=0.0,
        gripper_state=1.0,
        x_step_size=0.01,
        y_step_size=0.01,
        z_step_size=0.01,
    ):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.device = device
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.gripper_state = gripper_state
        
        self.joycon_robotics = None
        self.last_position = [0.0, 0.0, 0.0]
        self.last_gripper_state = gripper_state
        
        # Button states for episode control
        self.last_button_states = {}
        self.current_button_states = {}

    def start(self):
        """Start the joycon controller."""
        try:
            self.joycon_robotics = JoyconRobotics(
                device=self.device,
                gripper_open=self.gripper_open,
                gripper_close=self.gripper_close,
                gripper_state=self.gripper_state,
                lerobot=True,
            )
            self.running = True
            logging.info(f"Joycon {self.device} controller started successfully")
        except Exception as e:
            logging.error(f"Failed to start Joycon controller: {e}")
            self.running = False
            raise

    def stop(self):
        """Stop the joycon controller."""
        self.running = False
        if self.joycon_robotics:
            try:
                self.joycon_robotics.disconnect()
            except Exception as e:
                logging.warning(f"Error disconnecting Joycon: {e}")
            self.joycon_robotics = None

    def reset_joycon(self):
        """Reset joycon calibration."""
        if self.joycon_robotics:
            self.joycon_robotics.reset_joycon()

    def update(self):
        """Update controller state."""
        if not self.running or not self.joycon_robotics:
            return

        try:
            # Update joycon state
            self.joycon_robotics.update()
            
            # Update button states for episode control
            self.last_button_states = self.current_button_states.copy()
            self.current_button_states = self._get_button_states()
            
            # Check for episode control buttons
            self._check_episode_control()
            
        except Exception as e:
            logging.warning(f"Error updating Joycon state: {e}")

    def _get_button_states(self):
        """Get current button states."""
        if not self.joycon_robotics:
            return {}
        
        # Map common buttons - adjust based on your joycon implementation
        button_states = {}
        try:
            # These button mappings may need adjustment based on your joycon implementation
            button_states['plus'] = self.joycon_robotics.listen_button('plus')
            button_states['minus'] = self.joycon_robotics.listen_button('minus')
            button_states['home'] = self.joycon_robotics.listen_button('home')
            button_states['capture'] = self.joycon_robotics.listen_button('capture')
        except:
            pass
        
        return button_states

    def _check_episode_control(self):
        """Check for episode control button presses."""
        # Plus button for success
        if (self.current_button_states.get('plus', False) and 
            not self.last_button_states.get('plus', False)):
            self.episode_end_status = TeleopEvents.SUCCESS
        
        # Minus button for failure/rerecord
        if (self.current_button_states.get('minus', False) and 
            not self.last_button_states.get('minus', False)):
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        
        # Home button for intervention
        self.intervention_flag = self.current_button_states.get('home', False)

    def get_deltas(self):
        """Get movement deltas from joycon."""
        if not self.running or not self.joycon_robotics:
            return 0.0, 0.0, 0.0

        try:
            # Get current position from joycon
            current_position = self.joycon_robotics.get_control("euler_rad")[:3]
            
            # Calculate deltas
            delta_x = (current_position[0] - self.last_position[0]) * self.x_step_size
            delta_y = (current_position[1] - self.last_position[1]) * self.y_step_size
            delta_z = (current_position[2] - self.last_position[2]) * self.z_step_size
            
            # Update last position
            self.last_position = current_position.copy()
            
            # Clip deltas to reasonable range
            delta_x = math_clip(delta_x, -0.1, 0.1)
            delta_y = math_clip(delta_y, -0.1, 0.1)
            delta_z = math_clip(delta_z, -0.1, 0.1)
            
            return delta_x, delta_y, delta_z
            
        except Exception as e:
            logging.warning(f"Error getting deltas from Joycon: {e}")
            return 0.0, 0.0, 0.0

    def gripper_command(self):
        """Get gripper command from joycon."""
        if not self.running or not self.joycon_robotics:
            return "stay"

        try:
            # Get current gripper state
            current_gripper = self.joycon_robotics.get_control("euler_rad")[6]  # Assuming gripper is at index 6
            
            # Determine command based on change
            if abs(current_gripper - self.last_gripper_state) > 0.1:
                if current_gripper > self.last_gripper_state:
                    command = "open"
                else:
                    command = "close"
                self.last_gripper_state = current_gripper
                return command
            
            return "stay"
            
        except Exception as e:
            logging.warning(f"Error getting gripper command from Joycon: {e}")
            return "stay"


class JoyconControllerOptim(InputController):
    """Optimized Joycon controller with full 6DOF control."""

    def __init__(
        self,
        device="right",
        gripper_open=1.0,
        gripper_close=0.0,
        gripper_state=1.0,
        horizontal_stick_mode="y",
        close_y=False,
        limit_dof=False,
        glimit=None,
        offset_position_m=None,
        offset_euler_rad=None,
        euler_reverse=None,
        direction_reverse=None,
        dof_speed=None,
        rotation_filter_alpha_rate=1,
        common_rad=True,
        lerobot=False,
        pitch_down_double=False,
        without_rest_init=False,
        pure_xz=True,
        pure_x=True,
        pure_y=True,
        change_down_to_gripper=False,
        lowpassfilter_alpha_rate=1,
        x_step_size=0.02,
        y_step_size=0.02,
        z_step_size=0.02,
        yaw_step_size=0.05,
        pitch_step_size=0.05,
        roll_step_size=0.05,
    ):
        super().__init__(x_step_size, y_step_size, z_step_size)
        
        # Store all parameters
        self.device = device
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.gripper_state = gripper_state
        self.yaw_step_size = yaw_step_size
        self.pitch_step_size = pitch_step_size
        self.roll_step_size = roll_step_size
        
        # Default values for optional parameters
        if glimit is None:
            glimit = [
                [0.125, -0.4, 0.046, -3.1, -1.5, -1.57],
                [0.380, 0.4, 0.23, 3.1, 1.5, 1.57],
            ]
        if offset_position_m is None:
            offset_position_m = [0.0, 0.0, 0.0]
        if offset_euler_rad is None:
            offset_euler_rad = [0.0, 0.0, 0.0]
        if euler_reverse is None:
            euler_reverse = [1, 1, 1]
        if direction_reverse is None:
            direction_reverse = [1, 1, 1]
        if dof_speed is None:
            dof_speed = [1, 1, 1, 1, 1, 1]
        
        self.joycon_robotics = None
        self.last_control = [0.0] * 7  # x, y, z, roll, pitch, yaw, gripper
        
        # Button states for episode control
        self.last_button_states = {}
        self.current_button_states = {}
        
        # Store initialization parameters
        self.init_params = {
            'device': device,
            'gripper_open': gripper_open,
            'gripper_close': gripper_close,
            'gripper_state': gripper_state,
            'horizontal_stick_mode': horizontal_stick_mode,
            'close_y': close_y,
            'limit_dof': limit_dof,
            'glimit': glimit,
            'offset_position_m': offset_position_m,
            'offset_euler_rad': offset_euler_rad,
            'euler_reverse': euler_reverse,
            'direction_reverse': direction_reverse,
            'dof_speed': dof_speed,
            'rotation_filter_alpha_rate': rotation_filter_alpha_rate,
            'common_rad': common_rad,
            'lerobot': lerobot,
            'pitch_down_double': pitch_down_double,
            'without_rest_init': without_rest_init,
            'pure_xz': pure_xz,
            'pure_x': pure_x,
            'pure_y': pure_y,
            'change_down_to_gripper': change_down_to_gripper,
            'lowpassfilter_alpha_rate': lowpassfilter_alpha_rate,
        }

    def start(self):
        """Start the optimized joycon controller."""
        try:
            self.joycon_robotics = JoyconRobotics(**self.init_params)
            self.running = True
            logging.info(f"Optimized Joycon {self.device} controller started successfully")
        except Exception as e:
            logging.error(f"Failed to start optimized Joycon controller: {e}")
            self.running = False
            raise

    def stop(self):
        """Stop the optimized joycon controller."""
        self.running = False
        if self.joycon_robotics:
            try:
                self.joycon_robotics.disconnect()
            except Exception as e:
                logging.warning(f"Error disconnecting optimized Joycon: {e}")
            self.joycon_robotics = None

    def reset_joycon(self):
        """Reset joycon calibration."""
        if self.joycon_robotics:
            self.joycon_robotics.reset_joycon()

    def update(self):
        """Update controller state."""
        if not self.running or not self.joycon_robotics:
            return

        try:
            # Update joycon state
            self.joycon_robotics.update()
            
            # Update button states for episode control
            self.last_button_states = self.current_button_states.copy()
            self.current_button_states = self._get_button_states()
            
            # Check for episode control buttons
            self._check_episode_control()
            
        except Exception as e:
            logging.warning(f"Error updating optimized Joycon state: {e}")

    def _get_button_states(self):
        """Get current button states."""
        if not self.joycon_robotics:
            return {}
        
        button_states = {}
        try:
            # These button mappings may need adjustment based on your joycon implementation
            button_states['plus'] = self.joycon_robotics.listen_button('plus')
            button_states['minus'] = self.joycon_robotics.listen_button('minus')
            button_states['home'] = self.joycon_robotics.listen_button('home')
            button_states['capture'] = self.joycon_robotics.listen_button('capture')
        except:
            pass
        
        return button_states

    def _check_episode_control(self):
        """Check for episode control button presses."""
        # Plus button for success
        if (self.current_button_states.get('plus', False) and 
            not self.last_button_states.get('plus', False)):
            self.episode_end_status = TeleopEvents.SUCCESS
        
        # Minus button for failure/rerecord
        if (self.current_button_states.get('minus', False) and 
            not self.last_button_states.get('minus', False)):
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        
        # Home button for intervention
        self.intervention_flag = self.current_button_states.get('home', False)

    def get_deltas(self):
        """Get 6DOF movement deltas from joycon."""
        if not self.running or not self.joycon_robotics:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            # Get current control values (x, y, z, roll, pitch, yaw, gripper)
            current_control = self.joycon_robotics.get_control("euler_rad")
            
            # Calculate deltas for position
            delta_x = (current_control[0] - self.last_control[0]) * self.x_step_size
            delta_y = (current_control[1] - self.last_control[1]) * self.y_step_size
            delta_z = (current_control[2] - self.last_control[2]) * self.z_step_size
            
            # Calculate deltas for orientation
            delta_roll = (current_control[3] - self.last_control[3]) * self.roll_step_size
            delta_pitch = (current_control[4] - self.last_control[4]) * self.pitch_step_size
            delta_yaw = (current_control[5] - self.last_control[5]) * self.yaw_step_size
            
            # Update last control values
            self.last_control = current_control.copy()
            
            # Clip deltas to reasonable ranges
            delta_x = math_clip(delta_x, -0.1, 0.1)
            delta_y = math_clip(delta_y, -0.1, 0.1)
            delta_z = math_clip(delta_z, -0.1, 0.1)
            delta_roll = math_clip(delta_roll, -0.5, 0.5)
            delta_pitch = math_clip(delta_pitch, -0.5, 0.5)
            delta_yaw = math_clip(delta_yaw, -0.5, 0.5)
            
            return delta_x, delta_y, delta_z, delta_yaw, delta_pitch, delta_roll
            
        except Exception as e:
            logging.warning(f"Error getting 6DOF deltas from Joycon: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def gripper_command(self):
        """Get gripper command from joycon."""
        if not self.running or not self.joycon_robotics:
            return "stay"

        try:
            # Get current gripper state
            current_control = self.joycon_robotics.get_control("euler_rad")
            current_gripper = current_control[6]  # Gripper is at index 6
            
            # Determine command based on change
            if abs(current_gripper - self.last_control[6]) > 0.1:
                if current_gripper > self.last_control[6]:
                    return "open"
                else:
                    return "close"
            
            return "stay"
            
        except Exception as e:
            logging.warning(f"Error getting gripper command from optimized Joycon: {e}")
            return "stay"