from dataclasses import dataclass
from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass()
@dataclass()
class MetaQuest3Config(TeleoperatorConfig):
    ipaddress: str
    port: str 