from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
import time

rs_config = RealSenseCameraConfig(
serial_number_or_name='f1480368',
color_mode=ColorMode.RGB,
)
cam = RealSenseCamera(rs_config)

cam.connect()

for i in range(1000):
    time.sleep(0.5)
    image_data = cam.async_read()
    print(image_data.shape)
