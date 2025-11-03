一、项目简介
本项目旨在将 Berxel Hawk / P150E 工业网络相机 集成进 LeRobot框架中，实现通过 Python 接口直接从 Berxel SDK 采集 RGB 与
深度（Depth）数据，支持实时显示、异步采集、多模态保存等功能。
二、环境准备
系统环境要求
Ubuntu 22.04 / 24.04
Python ≥ 3.10
LeRobot ≥ 0.3.4
OpenCV ≥ 4.9.0
Berxel SDK v2.0.161（官方提供），如需图像显示，请安装 GUI 依赖：sudo apt install -y libgl1 libglib2.0-0 libgtk2.0-dev pkg-config
lerobot/BerxelSdkDriver/
文件结构如下
BerxelSdkDriver/
 ├── BerxelHawkNativeMethods.py
 ├── BerxelHawkDefines.py
 └── libs/
     ├── libBerxelInterface.so
     ├── libBerxelCommonDriver.so
     ├── libBerxelNetDriver.so
     ├── libBerxelLogDriver.so
     ├── libBerxelUvcDriver.so
     └── libBerxelHawk.so
三、在 LeRobot 中封装相机类
配置定义
创建文件 lerobot/src/lerobot/cameras/berxel/configuration_berxel.py
相机类实现
创建文件 lerobot/src/lerobot/cameras/berxel/berxel_camera.py：
该类主要包含：
SDK 初始化与连接 (berxelInit, berxelOpenDeviceByAddr)，彩色/深度帧采集与释放，异步读取线程
在 LeRobot 框架注册相机类型
四、运行测试脚本准备了两个测试脚本test1,test2,分别是连接相机观测深度图像跟彩色图像，测试二为保存十帧深度图像跟彩色图像为parquet文件以及图像。

