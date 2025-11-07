from dataclasses import dataclass

from draccus import field

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("meta_quest3")
@dataclass()
class MetaQuest3Config(TeleoperatorConfig):
    port: str
    ipaddress: str
    move_scale: float = 1.0
    rot_scale: float = 1.0
    hand_name: str = 'left'  # 'left', 'right', or 'both'