from lerobot.robots.ur5_follower.ur5_follower import UR5Follower
from lerobot.robots.ur5_follower.ur5_follower import UR5FollowerConfig
from lerobot.utils.utils import log_say

cfg=UR5FollowerConfig(robot_ip="192.168.1.20",with_gripper=True)

robot=UR5Follower(cfg)
if not robot.is_connected:
    robot.connect()
print("Connected to UR5 Follower robot.")
robot.robot.freedriveMode()
print("Robot is now in free drive mode. You can move it freely.")
log_say("机械臂可以随便掰了")
robot.disconnect()