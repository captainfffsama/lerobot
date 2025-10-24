import logging
import threading
import time
from typing import Any, Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation

from .meta_quest3_server import MetaQuest3Server
from ..teleoperator import Teleoperator
from .config_meta_quest3 import MetaQuest3Config, DualMetaQuest3Config
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

class MetaQuest3Teleop(Teleoperator):
    """
    Meta Quest 3 teleoperator for VR controller-based robot control.
    
    This teleoperator receives tracking data from Meta Quest 3 controllers
    and converts them into robot control actions. It supports 6DOF control
    with both hands and includes button/trigger inputs for gripper control.
    """
    
    config_class = MetaQuest3Config
    name = "meta_quest3"

    def __init__(self, config: MetaQuest3Config):
        super().__init__(config)
        self.config = config
        self._latest_tracking_data = None
        self._tracking_lock = threading.Lock()
        self._is_calibrated = False
        self._previous_poses = {}
        
        # Start server in a separate thread
        self._server_thread = None
        # Initialize server with reference to this teleoperator
        self.server = MetaQuest3Server(host=config.ipaddress, port=int(config.port), teleoperator=self)
        
        # Generate action names based on hand configuration
        if config.hand_name == 'both':
            self.action_names = (
                "l_delta_x", "l_delta_y", "l_delta_z", "l_delta_yaw", "l_delta_pitch", "l_delta_roll",
                "r_delta_x", "r_delta_y", "r_delta_z", "r_delta_yaw", "r_delta_pitch", "r_delta_roll"
            )
        else:
            self.action_names = ("delta_x", "delta_y", "delta_z", "delta_yaw", "delta_pitch", "delta_roll")

    @property
    def action_features(self) -> dict:
        """
        Define the action features structure for Meta Quest 3 controllers.
        Returns 6DOF pose data for both hands plus button/trigger inputs.
        """
        return {action_name : float for action_name in self.action_names}

    @property
    def feedback_features(self) -> dict:
        """
        Define feedback features - currently not supported by Meta Quest 3.
        """
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if Meta Quest 3 server is connected and receiving data."""
        return self.server.is_connected and self._latest_tracking_data is not None

    def connect(self, calibrate: bool = True) -> None:
        """Connect to Meta Quest 3 and start the server."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        logger.info("Starting Meta Quest 3 server...")
        
        # Start server in a separate thread
        self._server_thread = threading.Thread(target=self.server.start, daemon=True)
        self._server_thread.start()
        
        # Wait for server to start and receive some data
        timeout = 10.0  # 10 seconds timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.server.is_connected:
                # Wait for first tracking data
                time.sleep(1.0)
                if self._latest_tracking_data is not None:
                    break
            time.sleep(0.1)
        
        if not self.is_connected:
            raise RuntimeError("Failed to connect to Meta Quest 3 - no tracking data received")
        
        logger.info(f"{self} connected successfully")
        
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        """Check if the teleoperator is calibrated."""
        return self._is_calibrated

    def calibrate(self) -> None:
        """Calibrate the Meta Quest 3 controllers."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        if self._latest_tracking_data is not None:
            # Check if both hands have valid poses
            if self._has_valid_poses():
                # Store calibration data
                self._update_previous_poses(self._latest_tracking_data)
                self._is_calibrated = True
                logger.info("Meta Quest 3 calibration completed")
                return

    def configure(self) -> None:
        """Configure the Meta Quest 3 teleoperator."""
        # No additional configuration needed for Meta Quest 3
        raise NotImplementedError("Meta quest3 does not support configuration.")

    def get_action(self) -> dict[str, Any]:
        """Get current action from Meta Quest 3 controllers."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        if not self.is_calibrated:
            return {}
        
        with self._tracking_lock:
            if self._latest_tracking_data is None:
                return {}
            
            tracking_data = self._latest_tracking_data.copy()
        
        # Extract controller data
        action = self._extract_controller_actions(tracking_data)
        
        # Update previous poses for delta calculations
        self._update_previous_poses(tracking_data)
        
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to Meta Quest 3 - not currently supported."""
        # Meta Quest 3 doesn't support haptic feedback through this interface
        raise NotImplementedError("Meta Quest 3 doesn't support haptic feedback through this interface")

    def disconnect(self) -> None:
        """Disconnect from Meta Quest 3."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        logger.info("Disconnecting from Meta Quest 3...")
        self.server.stop()
        
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        
        self._latest_tracking_data = None
        self._is_calibrated = False
        self._previous_poses = {}
        
        logger.info(f"{self} disconnected")

    def _has_valid_poses(self) -> bool:
        """Check if both hands have valid pose data."""
        if self._latest_tracking_data is None:
            return False
        
        data = self._latest_tracking_data.get('data', {})
        controller_data = data.get('Controller', {})
        
        return 'left' in controller_data and 'right' in controller_data

    def _extract_controller_actions(self, tracking_data: dict) -> dict[str, Any]:
        """Extract controller actions from tracking data and convert to delta format."""
        if not tracking_data:
            return {name: 0.0 for name in self.action_names}
        
        current_pose = self.extract_quest_pose(tracking_data)
        
        if self.config.hand_name == 'both':
            # Handle both hands
            action = {}
            for hand in ['left', 'right']:
                prefix = 'l_' if hand == 'left' else 'r_'
                
                if hand in current_pose and hand in self._previous_poses:
                    hand_data = current_pose[hand]
                    prev_data = self._previous_poses[hand]
                    
                    trigger_pressed = hand_data['trigger'] > 0.5
                    
                    if trigger_pressed:
                        # Calculate position delta
                        pos_delta = hand_data['position'] - prev_data['position']
                        
                        # Calculate rotation delta
                        relative_rot = hand_data['rotation'] * prev_data['rotation'].inv()
                        rot_delta = relative_rot.as_euler('xyz', degrees=True)
                    else:
                        pos_delta = np.zeros(3)
                        rot_delta = np.zeros(3)
                    
                    # Apply scaling
                    pos_delta *= self.config.move_scale
                    rot_delta *= self.config.rot_scale
                    
                    action.update({
                        f'{prefix}delta_x': float(pos_delta[0]) if trigger_pressed else 0.0,
                        f'{prefix}delta_y': float(pos_delta[1]) if trigger_pressed else 0.0,
                        f'{prefix}delta_z': float(pos_delta[2]) if trigger_pressed else 0.0,
                        f'{prefix}delta_roll': float(rot_delta[0]) if trigger_pressed else 0.0,
                        f'{prefix}delta_pitch': float(rot_delta[1]) if trigger_pressed else 0.0,
                        f'{prefix}delta_yaw': float(rot_delta[2]) if trigger_pressed else 0.0,
                    })
                else:
                    # No data available for this hand
                    action.update({
                        f'{prefix}delta_x': 0.0,
                        f'{prefix}delta_y': 0.0,
                        f'{prefix}delta_z': 0.0,
                        f'{prefix}delta_roll': 0.0,
                        f'{prefix}delta_pitch': 0.0,
                        f'{prefix}delta_yaw': 0.0,
                    })
            
            return action
        else:
            # Handle single hand
            if not current_pose or self._previous_poses is None:
                return {name: 0.0 for name in self.action_names}
            
            trigger_pressed = current_pose['trigger'] > 0.5
            
            if trigger_pressed:
                # Calculate position delta
                pos_delta = current_pose['position'] - self._previous_poses['position']
                
                # Calculate rotation delta
                relative_rot = current_pose['rotation'] * self._previous_poses['rotation'].inv()
                rot_delta = relative_rot.as_euler('xyz', degrees=True)
            else:
                pos_delta = np.zeros(3)
                rot_delta = np.zeros(3)
            
            # Apply scaling
            pos_delta *= self.config.move_scale
            rot_delta *= self.config.rot_scale
            
            action = {
                'delta_x': float(pos_delta[0]) if trigger_pressed else 0.0,
                'delta_y': float(pos_delta[1]) if trigger_pressed else 0.0,
                'delta_z': float(pos_delta[2]) if trigger_pressed else 0.0,
                'delta_roll': float(rot_delta[0]) if trigger_pressed else 0.0,
                'delta_pitch': float(rot_delta[1]) if trigger_pressed else 0.0,
                'delta_yaw': float(rot_delta[2]) if trigger_pressed else 0.0,
            }
            
            return action

    def _update_previous_poses(self, tracking_data: dict):
        self._previous_poses = self.extract_quest_pose(tracking_data)

    def extract_quest_pose(self, tracking_data: dict) -> dict:
        """Extract pose data from current tracking data."""
        if not tracking_data:
            return {}

        controller_data = tracking_data.get('data', {}).get('Controller', {})
        
        if self.config.hand_name == 'both':
            # Extract both hands data
            result = {}
            for hand in ['left', 'right']:
                if hand in controller_data:
                    hand_data = controller_data[hand]
                    pose = hand_data.get('parsed_pose', {})
                    
                    position = np.array([
                        pose.get('position', {}).get('x', 0.0),
                        pose.get('position', {}).get('y', 0.0),
                        pose.get('position', {}).get('z', 0.0)
                    ])
                    
                    rotation = np.array([
                        pose.get('rotation', {}).get('x', 0.0),
                        pose.get('rotation', {}).get('y', 0.0),
                        pose.get('rotation', {}).get('z', 0.0),
                        pose.get('rotation', {}).get('w', 1.0)
                    ])
                    
                    result[hand] = {
                        'position': position,
                        'rotation': Rotation.from_quat(rotation),
                        'trigger': hand_data.get('trigger', 0.0)
                    }
            return result
        else:
            # Extract single hand data
            if self.config.hand_name not in controller_data:
                return {}
                
            hand_data = controller_data[self.config.hand_name]
            pose = hand_data.get('parsed_pose', {})
            
            position = np.array([
                pose.get('position', {}).get('x', 0.0),
                pose.get('position', {}).get('y', 0.0),
                pose.get('position', {}).get('z', 0.0)
            ])
            
            rotation = np.array([
                pose.get('rotation', {}).get('x', 0.0),
                pose.get('rotation', {}).get('y', 0.0),
                pose.get('rotation', {}).get('z', 0.0),
                pose.get('rotation', {}).get('w', 1.0)
            ])
            
            return {
                'position': position,
                'rotation': Rotation.from_quat(rotation),
                'trigger': hand_data.get('trigger', 0.0)
            }

    def update_tracking_data(self, tracking_data: dict):
        """Update the latest tracking data (called by server)."""
        with self._tracking_lock:
            self._latest_tracking_data = tracking_data
