from lerobot.robots.ur5_follower.ur5_follower import UR5Follower
from lerobot.robots.ur5_follower.ur5_follower import UR5FollowerConfig
from lerobot.utils.utils import log_say

cfg=UR5FollowerConfig(robot_ip="192.168.1.20",with_gripper=True)

robot=UR5Follower(cfg)
if not robot.is_connected:
    robot.connect(calibrate=False)
print("Connected to UR5 Follower robot.")
robot.robot.freedriveMode()
print("Robot is now in free drive mode. You can move it freely.")
log_say("While other robots dream of exploring Mars, the Rega robot's sole ambition is to conquer surface noise on a 180-gram vinyl record")
robot.disconnect()