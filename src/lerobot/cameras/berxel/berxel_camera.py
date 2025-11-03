import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
import ctypes
import sys
import os

from ...utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_berxel import BerxelCameraConfig
from contextlib import contextmanager

# 导入 Berxel SDK
SDK_PATH = "/home/majinda/lerobot/BerxelSdkDriver"
sys.path.append(SDK_PATH)
from BerxelHawkNativeMethods import *
from BerxelHawkDefines import *

logger = logging.getLogger(__name__)


class BerxelCamera(Camera):
    """
    适配 Berxel Hawk / P150E 网络相机的 RGB + Depth 采集接口。
    兼容 LeRobot 的 Camera 抽象层，可直接用于 dataset 采集。
    """
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        扫描网络上可用的 Berxel 相机。
        （目前简单返回一个固定 IP，也可以以后扩展成自动发现）
        """
        return [{
            "type": "Berxel",
            "id": "192.168.2.11",
            "name": "Berxel Hawk P150E",
            "ip": "192.168.2.11"
        }]

    def __init__(self, config: BerxelCameraConfig):
        super().__init__(config)
        self.config = config
        self.ip = config.ip.encode("utf-8")

        self.stream_color = streamHandle()
        self.stream_depth = streamHandle()
        self.device = deviceHandle()

        self.use_depth = config.use_depth
        self.fps = config.fps or 30
        self.width = config.width or 1280
        self.height = config.height or 720

        self.rotation = get_cv2_rotation(config.rotation)
        self.color_mode = config.color_mode

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()
    @contextmanager
    def suppress_c_logs(self):
        """静音 C 层 printf 输出"""
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_out, save_err = os.dup(1), os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        try:
            yield
        finally:
            os.dup2(save_out, 1)
            os.dup2(save_err, 2)
            os.close(null_fd)

    @property
    def is_connected(self) -> bool:
        return bool(self.device)

    def connect(self, warmup=True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Berxel camera already connected")

        logger.info("🚀 初始化 Berxel SDK ...")

        with self.suppress_c_logs():
            ret = berxelInit()
            if ret != 0:
                raise RuntimeError(f"SDK 初始化失败: {ret}")

            ret = berxelOpenDeviceByAddr(self.ip, ctypes.byref(self.device))
            if ret != 0:
                raise ConnectionError(f"打开设备失败: {ret}")

            ret = berxelOpenStream(self.device, BERXEL_HAWK_COLOR_STREAM, ctypes.byref(self.stream_color))
            if ret != 0:
                raise RuntimeError(f"打开彩色流失败: {ret}")

            if self.use_depth:
                ret = berxelOpenStream(self.device, BERXEL_HAWK_DEPTH_STREAM, ctypes.byref(self.stream_depth))
                if ret != 0:
                    logger.warning("⚠️ 深度流打开失败，仅使用彩色流")

        if warmup:
            time.sleep(1.0)
            for _ in range(5):
                _ = self.read()

        logger.info("✅ Berxel 相机连接成功")

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 500) -> np.ndarray:
        """读取一帧彩色图像"""
        frm = imageFrameHandle()
        ret = berxelReadFrame(self.stream_color, ctypes.byref(frm), timeout_ms)
        if ret != 0:
            raise RuntimeError(f"读取彩色帧失败: {ret}")

        f = frm.contents
        buf = ctypes.string_at(f.pVoidData, f.dataSize)
        img = np.frombuffer(buf, dtype=np.uint8).reshape((f.height, f.width, 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        berxelReleaseFrame(ctypes.byref(frm))
        return img

    def read_depth(self, timeout_ms: int = 500) -> np.ndarray:
        """读取一帧深度图"""
        if not self.use_depth:
            raise RuntimeError("深度流未启用")

        frm = imageFrameHandle()
        ret = berxelReadFrame(self.stream_depth, ctypes.byref(frm), timeout_ms)
        if ret != 0:
            raise RuntimeError(f"读取深度帧失败: {ret}")

        f = frm.contents
        buf = ctypes.string_at(f.pVoidData, f.dataSize)
        depth = np.frombuffer(buf, dtype=np.uint16).reshape((f.height, f.width))
        berxelReleaseFrame(ctypes.byref(frm))
        return depth

    def async_read(self, timeout_ms: int = 500) -> np.ndarray:
        """异步读取最新彩色帧（兼容 LeRobot pipeline）"""
        if not self.thread or not self.thread.is_alive():
            self._start_thread()

        if not self.new_frame_event.wait(timeout_ms / 1000):
            raise TimeoutError("等待帧超时")

        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError("未获取到有效帧数据")
        return frame

    def _start_thread(self):
        """启动异步读取线程"""
        self.stop_event = Event()
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        """循环抓取帧"""
        while not self.stop_event.is_set():
            try:
                img = self.read(timeout_ms=500)
                with self.frame_lock:
                    self.latest_frame = img
                self.new_frame_event.set()
            except Exception as e:
                logger.warning(f"读取线程错误: {e}")
                time.sleep(0.1)

    def disconnect(self):
        """关闭设备与释放资源"""
        with self.suppress_c_logs():
            if self.stop_event:
                self.stop_event.set()

            if self.stream_color:
                berxelCloseStream(self.stream_color)
            if self.stream_depth:
                berxelCloseStream(self.stream_depth)
            if self.device:
                berxelCloseDevice(self.device)
            berxelDestroy()
        logger.info("✅ Berxel 相机已断开连接")
