import redis
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation


class X1RedisFollower:
    """
    Base class for controlling X1 robot arm via Redis
    """

    def __init__(self, host="127.0.0.1", port=6379, armID=1):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.armID = armID
        self.move_params = {
            "velocity": 0.05,
            "acc": 5,
            "jerk": 50,
            "dt": 0.01
        }
        self.connected = False

    def _gen_taskid(self):
        return int(time.time() * 1000)

    def _send(self, key, action, params=None):
        params = params or {}
        payload = {
            "taskID": self._gen_taskid(),
            "armID": self.armID,
            "action": action,
            "parsms": params,
        }
        self.r.set(key, json.dumps(payload))
        return payload["taskID"]

    # === 基础控制 ===
    def connect(self):
        """连接机器人"""
        self._send("redis_robot_base", 0)
        self.connected = True
        print("✅ X1 已连接")

    def enable(self):
        self._send("redis_robot_base", 1)
        print("⚡ 上使能")

    def disable(self):
        self._send("redis_robot_base", 2)
        print("🛑 下使能")

    # === 读取状态 ===
    def get_joint_angles(self):
        self._send("redis_robot_base", 4)
        time.sleep(0.05)
        ans = self.r.get("redis_robot_base_ans")
        if ans:
            data = json.loads(ans)
            return data["parsms"].get("left_joints") or data["parsms"].get("right_joint")
        return None

    def get_eef_pose(self):
        self._send("redis_robot_base", 5)
        time.sleep(0.05)
        ans = self.r.get("redis_robot_base_ans")
        if ans:
            data = json.loads(ans)
            return data["parsms"].get("left_pose") or data["parsms"].get("right_pose")
        return None

    # === 控制命令 ===
    def moveJ(self, joints):
        """绝对关节控制"""
        params = {**self.move_params,
                  "left_joints": joints}
        self._send("redis_robot_move", 20, params)

    def moveL(self, pose):
        """绝对末端控制"""
        params = {**self.move_params,
                  "left_pose": pose}
        self._send("redis_robot_move", 22, params)

    def moveL_delta(self, delta_pose):
        """相对末端控制"""
        cur_pose = self.get_eef_pose()
        if not cur_pose:
            print("❌ 无法获取末端位姿")
            return
        R_cur = Rotation.from_quat(cur_pose[3:])
        R_delta = Rotation.from_euler("xyz", delta_pose[3:], degrees=False)
        R_new = R_cur * R_delta
        quat_new = R_new.as_quat()
        pose_new = [
            cur_pose[0] + delta_pose[0],
            cur_pose[1] + delta_pose[1],
            cur_pose[2] + delta_pose[2],
            *quat_new
        ]
        params = {**self.move_params, "left_pose": pose_new}
        self._send("redis_robot_move", 23, params)
