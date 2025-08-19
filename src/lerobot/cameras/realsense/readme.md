## L515 安装注意事项

### 1、SDK安装

```
安装支持L515的RealSense SDK 2.0，目前SDK版本测试，V2.49.0，可以正常使用。注意：V2.56.3不可以使用。

源码编译安装过程，请参考如下链接：
https://github.com/IntelRealSense/librealsense/blob/development/doc/installation.md

```

### 2、固件（Firmware）安装

```
固件（Firmware）安装 1.5.8.1版本
```

3、pyrealsense2依赖包安装

```
安装pyrealsense2依赖包，版本是V2.54.1.5216，注意V2.56系列版本不可以使用。
```

### 3. 如何查看realsense支持的分辨率和帧率
```python
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) > 0:
    dev = devices[0] # 获取第一个设备
    sensors = dev.query_sensors() # 获取所有传感器

    for sensor in sensors:
        print(f"--- 传感器: {sensor.get_info(rs.camera_info.name)} ---")
        for profile in sensor.get_stream_profiles():
            # profile.as_video_stream_profile() 用于过滤出视频流
            if profile.is_video_stream_profile():
                v_profile = profile.as_video_stream_profile()
                print(f"  {v_profile.stream_name()}: {v_profile.width()}x{v_profile.height()} @ {v_profile.fps()}Hz, Format: {v_profile.format()}")
else:
    print("未找到RealSense设备。")
```