from dataclasses import dataclass

@dataclass
class X1FollowerConfig:
    """
    Config for X1RedisFollower
    """
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    armID: int = 1  # 1左臂，2右臂，0双臂
    control_mode: str = "eef"  # 可选: 'eef', 'joint'
    velocity: float = 0.05
    acc: float = 5.0
    jerk: float = 50.0
    dt: float = 0.01
