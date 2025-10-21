import logging
import threading
import time
from typing import Any, Dict, Optional
import numpy as np

from .meta_quest3_server import MetaQuest3Server
from ..teleoperator import Teleoperator
from .config_meta_quest3 import MetaQuest3Config
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
        self._calibration_data = {}
        self._previous_poses = {}
        
        # Start server in a separate thread
        self._server_thread = None
        # Initialize server with reference to this teleoperator
        self.server = MetaQuest3Server(host=config.ipaddress, port=int(config.port), teleoperator=self)

    @property
    def action_features(self) -> dict:
        """
        Define the action features structure for Meta Quest 3 controllers.
        Returns 6DOF pose data for both hands plus button/trigger inputs.
        """
        return {
            "left_hand.pos": np.ndarray,  # shape (3,) - left hand position
            "left_hand.rot": np.ndarray,  # shape (4,) - left hand quaternion
            "left_hand.grip": float,      # left hand grip value [0, 1]
            "left_hand.trigger": float,  # left hand trigger value [0, 1]
            "left_hand.buttons": dict,   # left hand button states
            
            "right_hand.pos": np.ndarray,  # shape (3,) - right hand position  
            "right_hand.rot": np.ndarray,  # shape (4,) - right hand quaternion
            "right_hand.grip": float,      # right hand grip value [0, 1]
            "right_hand.trigger": float,  # right hand trigger value [0, 1]
            "right_hand.buttons": dict,   # right hand button states
            
            "enabled": bool,  # whether teleoperation is active
        }

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
        
        logger.info("Calibrating Meta Quest 3 controllers...")
        logger.info("Please hold both controllers in a neutral position and press the grip buttons")
        
        # Wait for calibration data
        calibration_timeout = 30.0  # 30 seconds
        start_time = time.time()
        
        while time.time() - start_time < calibration_timeout:
            if self._latest_tracking_data is not None:
                # Check if both hands have valid poses
                if self._has_valid_poses():
                    # Store calibration data
                    self._calibration_data = self._extract_calibration_data()
                    self._is_calibrated = True
                    logger.info("Meta Quest 3 calibration completed")
                    return
            time.sleep(0.1)
        
        raise RuntimeError("Calibration timeout - no valid tracking data received")

    def configure(self) -> None:
        """Configure the Meta Quest 3 teleoperator."""
        # No additional configuration needed for Meta Quest 3
        pass

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
        pass

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
        self._calibration_data = {}
        self._previous_poses = {}
        
        logger.info(f"{self} disconnected")

    def _has_valid_poses(self) -> bool:
        """Check if both hands have valid pose data."""
        if self._latest_tracking_data is None:
            return False
        
        data = self._latest_tracking_data.get('data', {})
        controller_data = data.get('Controller', {})
        
        return 'left' in controller_data and 'right' in controller_data

    def _extract_calibration_data(self) -> dict:
        """Extract calibration data from current tracking data."""
        if self._latest_tracking_data is None:
            return {}
        
        data = self._latest_tracking_data.get('data', {})
        controller_data = data.get('Controller', {})
        
        calibration = {}
        for hand in ['left', 'right']:
            if hand in controller_data:
                hand_data = controller_data[hand]
                if 'parsed_pose' in hand_data:
                    pose = hand_data['parsed_pose']
                    calibration[hand] = {
                        'position': np.array([
                            pose['position']['x'],
                            pose['position']['y'], 
                            pose['position']['z']
                        ]),
                        'rotation': np.array([
                            pose['rotation']['x'],
                            pose['rotation']['y'],
                            pose['rotation']['z'],
                            pose['rotation']['w']
                        ])
                    }
        
        return calibration

    def _extract_controller_actions(self, tracking_data: dict) -> dict[str, Any]:
        """Extract controller actions from tracking data."""
        if not tracking_data:
            return {}
        
        data = tracking_data.get('data', {})
        controller_data = data.get('Controller', {})
        
        action = {}
        
        # Process each hand
        for hand in ['left', 'right']:
            if hand in controller_data:
                hand_data = controller_data[hand]
                
                # Extract pose data
                if 'parsed_pose' in hand_data:
                    pose = hand_data['parsed_pose']
                    position = np.array([
                        pose['position']['x'],
                        pose['position']['y'],
                        pose['position']['z']
                    ])
                    rotation = np.array([
                        pose['rotation']['x'],
                        pose['rotation']['y'], 
                        pose['rotation']['z'],
                        pose['rotation']['w']
                    ])
                    
                    # Apply calibration if available
                    if self._is_calibrated and hand in self._calibration_data:
                        calib = self._calibration_data[hand]
                        position = position - calib['position']
                        # Note: Rotation calibration would require quaternion math
                
                else:
                    position = np.zeros(3)
                    rotation = np.array([0, 0, 0, 1])  # Identity quaternion
                
                # Extract input values
                grip = hand_data.get('grip', 0.0)
                trigger = hand_data.get('trigger', 0.0)
                
                # Extract button states
                buttons = {
                    'primaryButton': hand_data.get('primaryButton', False),
                    'secondaryButton': hand_data.get('secondaryButton', False),
                    'menuButton': hand_data.get('menuButton', False),
                    'axisClick': hand_data.get('axisClick', False),
                }
                
                # Store in action dict
                action[f"{hand}_hand.pos"] = position
                action[f"{hand}_hand.rot"] = rotation
                action[f"{hand}_hand.grip"] = float(grip)
                action[f"{hand}_hand.trigger"] = float(trigger)
                action[f"{hand}_hand.buttons"] = buttons
        
        # Determine if teleoperation is enabled (both grips pressed)
        left_grip = action.get('left_hand.grip', 0.0)
        right_grip = action.get('right_hand.grip', 0.0)
        action['enabled'] = left_grip > 0.5 and right_grip > 0.5
        
        return action

    def _update_previous_poses(self, tracking_data: dict):
        """Update previous poses for delta calculations."""
        if not tracking_data:
            return
        
        data = tracking_data.get('data', {})
        controller_data = data.get('Controller', {})
        
        for hand in ['left', 'right']:
            if hand in controller_data:
                hand_data = controller_data[hand]
                if 'parsed_pose' in hand_data:
                    pose = hand_data['parsed_pose']
                    self._previous_poses[hand] = {
                        'position': np.array([
                            pose['position']['x'],
                            pose['position']['y'],
                            pose['position']['z']
                        ]),
                        'rotation': np.array([
                            pose['rotation']['x'],
                            pose['rotation']['y'],
                            pose['rotation']['z'],
                            pose['rotation']['w']
                        ])
                    }

    def update_tracking_data(self, tracking_data: dict):
        """Update the latest tracking data (called by server)."""
        with self._tracking_lock:
            self._latest_tracking_data = tracking_data

