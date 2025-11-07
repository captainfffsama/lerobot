from dataclasses import dataclass, field
from typing import Optional
import logging

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseTeleoperatorConfig(TeleoperatorConfig):
    device_path: Optional[str] = None  # None = 自动检测
    use_gripper: bool = False  # 是否启用夹爪按钮

    scale_translation: float = 1.0
    scale_rotation: float = 1.0
    deadzone: float = 0.02

    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_roll: bool = False
    invert_pitch: bool = False
    invert_yaw: bool = False


@TeleoperatorConfig.register_subclass("bispacemouse")
@dataclass
class BiSpacemouseTeleopConfig(TeleoperatorConfig):
    left_device_path: Optional[str] = None  # None = 自动检测
    right_device_path: Optional[str] = None  # None = 自动检测
    left_spacemouse_config: SpacemouseTeleoperatorConfig = field(default_factory=SpacemouseTeleoperatorConfig)
    right_spacemouse_config: SpacemouseTeleoperatorConfig = field(
        default_factory=SpacemouseTeleoperatorConfig
    )

    def __post_init__(self):
        self.left_spacemouse_config.id = "left"
        self.right_spacemouse_config.id = "right"
        if self.left_device_path is None or self.right_device_path is None:
            from easyhid import Enumeration
            from pyspacemouse.pyspacemouse import device_specs

            hid = Enumeration()
            all_hids = hid.find()
            available_devices = []
            for device in all_hids:
                for name, spec in device_specs.items():
                    if device.vendor_id == spec.hid_id[0] and device.product_id == spec.hid_id[1]:
                        available_devices.append(device.path)
            self.left_device_path, self.right_device_path = assign_devices(
                [self.left_device_path, self.right_device_path],
                available_devices,
            )
        self.left_spacemouse_config.device_path = self.left_device_path
        self.right_spacemouse_config.device_path = self.right_device_path
        logging.info(f"Left Spacemouse device: {self.left_device_path}")
        logging.info(f"Right Spacemouse device: {self.right_device_path}")


def assign_devices(A: list, B: list) -> list:
    if None not in A:
        return list(A)
    used_devices = {device for device in A if device is not None}
    available_pool = set(B) - used_devices
    available_iter = iter(available_pool)
    C = []
    for device in A:
        if device is not None:
            C.append(device)
        else:
            try:
                C.append(next(available_iter))
            except StopIteration:
                raise ValueError(f"list {B} does not have enough available devices")
    return C
