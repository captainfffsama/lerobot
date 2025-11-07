import threading
import time
from typing import Any, Dict

try:
    import pyspacemouse
except ImportError:  # 允许延迟导入（便于无设备环境跑单测）
    pyspacemouse = None

from lerobot.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError

from ..teleoperator import Teleoperator
from .config_spacemouse import SpacemouseTeleoperatorConfig, BiSpacemouseTeleopConfig

AXES_ALL = ["x", "y", "z", "roll", "pitch", "yaw"]


class SpacemouseTeleoperator(Teleoperator):
    name = "spacemouse"
    config_class = SpacemouseTeleoperatorConfig

    def __init__(self, config: SpacemouseTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._last_state = None
        self._last_time = None

        self.spacemouse = None  # type: ignore
        self.action_names = (
            ("delta_x", "delta_y", "delta_z", "delta_yaw", "delta_pitch", "delta_roll", "gripper")
            if self.config.use_gripper
            else ("delta_x", "delta_y", "delta_z", "delta_yaw", "delta_pitch", "delta_roll")
        )
        self._mouse_data = None
        self._gripper_open = True  # 夹爪初始状态
        self._previous_data = None

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(self.action_names, float)

    def connect(self, calibrate: bool = True) -> None:
        if pyspacemouse is None:
            raise RuntimeError("pyspacemouse not installed")
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} already connected")
        self.spacemouse = pyspacemouse.open(path=self.config.device_path)
        if self.spacemouse is None:
            raise DeviceNotConnectedError(f"SpaceMouse device {self.config.device_path} not connected")
        self._connected = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def _update_loop(self):
        while self.is_connected:
            try:
                st = self.spacemouse.read()
                self._mouse_data = {
                    "t": st.t,
                    "x": st.x,
                    "y": st.y,
                    "z": st.z,
                    "roll": st.roll,
                    "pitch": st.pitch,
                    "yaw": st.yaw,
                    "lb": st.buttons[0],
                    "rb": st.buttons[1],
                }
                if not self._previous_data:
                    self._previous_data = self._mouse_data
            except Exception as e:
                print(f"Error reading SpaceMouse data: {e}")
            time.sleep(0.01)  # 避免过高频率读取

    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        if self.is_connected and pyspacemouse is not None:
            try:
                self.spacemouse.close()
            except Exception:
                self.spacemouse = None
        self._connected = False

    def calibrate(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict) -> None:
        pass

    @property
    def feedback_features(self) -> dict:
        return {}

    def get_action(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("SpaceMouse device not connected, cannot get action")

        action_dict = dict.fromkeys(self.action_features.keys(), 0.0)
        for axis in AXES_ALL:
            data = self._mouse_data[axis]
            data = self._apply_invert(axis, data)
            data = self._apply_scale(axis, data)
            data = self._apply_deadzone(data)
            action_dict[f"delta_{axis}"] = data

        if self.config.use_gripper:
            lb = self._mouse_data.get("lb", 0)
            if lb and self._previous_data["lb"] != lb:
                self._gripper_open = not self._gripper_open
            action_dict["gripper"] = 1.0 if self._gripper_open else 0.0
        return action_dict

    def _apply_deadzone(self, v: float) -> float:
        if abs(v) < self.config.deadzone:
            return 0.0
        return v

    def _apply_scale(self, axis: str, v: float) -> float:
        if axis in ("x", "y", "z"):
            return v * self.config.scale_translation
        else:
            return v * self.config.scale_rotation

    def _apply_invert(self, axis: str, v: float) -> float:
        cfg = self.config
        if axis == "x" and cfg.invert_x:
            return -v
        if axis == "y" and cfg.invert_y:
            return -v
        if axis == "z" and cfg.invert_z:
            return -v
        if axis == "roll" and cfg.invert_roll:
            return -v
        if axis == "pitch" and cfg.invert_pitch:
            return -v
        if axis == "yaw" and cfg.invert_yaw:
            return -v
        return v


class BiSpacemouseTeleop(Teleoperator):
    config_class = BiSpacemouseTeleopConfig
    name = "bispacemouse"

    def __init__(self, config: BiSpacemouseTeleopConfig):
        super().__init__(config)
        self.left_spacemouse = SpacemouseTeleoperator(config.left_spacemouse_config)
        self.right_spacemouse = SpacemouseTeleoperator(config.right_spacemouse_config)
        self.action_names = tuple(
            [f"l_{name}" for name in self.left_spacemouse.action_names]
            + [f"r_{name}" for name in self.right_spacemouse.action_names]
        )

    def connect(self, calibrate: bool = True) -> None:
        """Connect to both SpaceMouse devices."""
        if self.left_spacemouse.is_connected:
            raise DeviceAlreadyConnectedError(f"Left {self.name} already connected")
        if self.right_spacemouse.is_connected:
            raise DeviceAlreadyConnectedError(f"Right {self.name} already connected")

        self.left_spacemouse.connect(calibrate=False)
        self.right_spacemouse.connect(calibrate=False)

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """Calibrate both SpaceMouse devices."""
        self.left_spacemouse.calibrate()
        self.right_spacemouse.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self.left_spacemouse.is_calibrated and self.right_spacemouse.is_calibrated

    @property
    def feedback_features(self) -> dict:
        raise NotImplementedError("BiSpacemouseTeleop does not support feedback.")

    def configure(self) -> None:
        """Configure the SpaceMouse devices."""
        pass  # No additional configuration needed

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the SpaceMouse devices."""
        raise NotImplementedError("BiSpacemouseTeleop does not support feedback.")

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(self.action_names, float)

    def get_action(self) -> dict:
        """Get actions from both SpaceMouse devices."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} devices are not connected.")

        l_action = self.left_spacemouse.get_action()
        r_action = self.right_spacemouse.get_action()
        action_dict = {f"l_{k}": v for k, v in l_action.items()}
        for k, v in r_action.items():
            action_dict[f"r_{k}"] = v
        return action_dict

    def disconnect(self):
        """Disconnect both SpaceMouse devices."""
        if self.is_connected:
            self.left_spacemouse.disconnect()
            self.right_spacemouse.disconnect()
        self.left_spacemouse = None
        self.right_spacemouse = None

    @property
    def is_connected(self) -> bool:
        return self.left_spacemouse.is_connected and self.right_spacemouse.is_connected
