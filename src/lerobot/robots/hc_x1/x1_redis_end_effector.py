import numpy as np
from scipy.spatial.transform import Rotation
from .x1_redis_follower import X1RedisFollower


class X1RedisEndEffector(X1RedisFollower):
    """
    End-effector control interface for X1 Robot (LeRobot-compatible)
    """

    def __init__(self, host="127.0.0.1", port=6379, armID=1):
        super().__init__(host, port, armID)

        self.motors_names = ("ee_x", "ee_y", "ee_z", "roll", "pitch", "yaw")
        self.action_names = ("delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw")

    def send_action(self, action_dict):
        """
        输入增量控制（如 VR 手柄遥操作 或 模型预测的 Δx, Δy, Δz, Δroll...）
        """
        delta = np.array([action_dict[name] for name in self.action_names])
        self.moveL_delta(delta)
        return True

    def command_eef_pos(self, eef_pos):
        """
        输入绝对末端目标位姿
        """
        self.moveL(eef_pos)
        return True

    def get_observation(self):
        """
        获取当前末端位姿（给上层RL/VLA输入）
        """
        pose = self.get_eef_pose()
        if pose is None:
            pose = [0, 0, 0, 0, 0, 0, 1]
        return dict(zip(self.motors_names, pose))
