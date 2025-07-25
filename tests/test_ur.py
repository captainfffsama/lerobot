from lerobot.robots.ur_follower import URFollower
from lerobot.robots.ur_follower import URFollowerConfig
from lerobot.teleoperators.ur_leader import URLeader
from lerobot.teleoperators.ur_leader import URLeaderConfig

NB_CYCLES_CLIENT_CONNECTION = 250
def test_ur():
    leader_arm_config = URLeaderConfig()
    leader_arm=URLeader(leader_arm_config)
    leader_arm.connect()

    while i < NB_CYCLES_CLIENT_CONNECTION:
        arm_action = leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}




        task = "Dummy Example Task Dataset"

        i += 1

    print("Disconnecting Teleop Devices and LeKiwi Client")
    leader_arm.disconnect()