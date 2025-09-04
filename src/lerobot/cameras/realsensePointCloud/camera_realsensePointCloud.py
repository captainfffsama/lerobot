#!/usr/bin/env python3
import threading
import cv2
import numpy as np
from collections import deque 
import imageio
import pyrealsense2 as rs
from multiprocessing import Process, Pipe, Queue, Event
import time
import multiprocessing
import torch
import pytorch3d.ops as torch3d_ops

from ..camera import Camera
from .configuration_camera_realsensePointCloud import RealsensePointCloudCameraConfig


multiprocessing.set_start_method('fork')

np.printoptions(3, suppress=True)

def get_realsense_id():
    ctx = rs.context()
    devices = ctx.query_devices()
    devices = [devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))]
    devices.sort() # Make sure the order is correct
    print("Found {} devices: {}".format(len(devices), devices))
    return devices

def init_given_realsense_L515(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        # L515
        h, w = 768, 1024
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        # L515
        h, w = 540, 960
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        # Set the inter-camera sync mode
        # Use 1 for master, 2 for slave, 0 for default (no sync)
        # for L515
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
        
        # set min distance
        # for L515
        depth_sensor.set_option(rs.option.min_distance, 0.05)
        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None

def init_given_realsense_D455(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        
        # D455
        # h, w = 720, 1280
        h, w = 480, 640
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        
        # h, w = 720, 1280
        h, w = 480, 640
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None

def init_given_realsense_D435(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        
        # D435
        h, w = 480, 640
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        
        h, w = 480, 640
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None




def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                   The first 3 columns represent the coordinates (x, y, z).
                   The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    coords = point_cloud[:, :3]  # Extract coordinates
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)
    
    # Create unique grid keys
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    
    # Select unique points based on grid keys
    _, indices = np.unique(keys, return_index=True)
    
    # Return sampled points
    return point_cloud[indices]


class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale = 1) :
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale
        
class SingleVisionProcess(Process):
    def __init__(self, device_name, serial_number, queue, box_bounds,
                enable_rgb=True,
                enable_depth=False,
                enable_pointcloud=False,
                sync_mode=0,
                num_points=2048,
                use_grid_sampling=True,
                use_crop=False,
                img_size=384) -> None:
        super(SingleVisionProcess, self).__init__()
        self.queue = queue
        self.device_name = device_name
        self.serial_number = serial_number

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_pointcloud = enable_pointcloud
        self.sync_mode = sync_mode
            
        self.use_grid_sampling = use_grid_sampling
        self.use_crop = use_crop

  
        self.resize = False
        # self.height, self.width = 512, 512
        self.height, self.width = img_size, img_size
        
        # point cloud params
        self.num_points = num_points
        self.box_bounds = box_bounds
   
    def get_vision(self):
        frame = self.pipeline.wait_for_frames()

        if self.enable_depth:
            aligned_frames = self.align.process(frame)
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
    
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
            
            clip_lower =  0.01
            clip_high = 1.0
            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale
            depth_frame[depth_frame < clip_lower] = clip_lower
            depth_frame[depth_frame > clip_high] = clip_high
            
            if self.enable_pointcloud:
                # Nx3
                point_cloud_frame = self.create_colored_point_cloud(color_frame, depth_frame, self.box_bounds,
                                    num_points=self.num_points, use_crop=self.use_crop)
            else:
                point_cloud_frame = None
        else:
            color_frame = frame.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = None
            point_cloud_frame = None

        # print("color:", color_frame.shape)
        # print("depth:", depth_frame.shape)
        
        if self.resize:
            if self.enable_rgb:
                color_frame = cv2.resize(color_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.enable_depth:
                depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return color_frame, depth_frame, point_cloud_frame


    def run(self):
        if self.device_name == "L515":
            init_given_realsense = init_given_realsense_L515
        elif self.device_name == "D435":
            init_given_realsense = init_given_realsense_D435
            print("Initializing realsense D435!")
        elif self.device_name == "D455":
            init_given_realsense = init_given_realsense_D455
        
        self.pipeline, self.align, self.depth_scale, self.camera_info = init_given_realsense(self.serial_number, 
                    enable_rgb=self.enable_rgb, enable_depth=self.enable_depth,
                    enable_point_cloud=self.enable_pointcloud,
                    sync_mode=self.sync_mode)

        debug = False
        while True:
            color_frame, depth_frame, point_cloud_frame = self.get_vision()
            self.queue.put([color_frame, depth_frame, point_cloud_frame])
            time.sleep(0.05)

    def terminate(self) -> None:
        # self.pipeline.stop()
        return super().terminate()

    def crop_point_cloud(self, point_cloud):
        # Nx3
        pass
    def create_colored_point_cloud(self, color, depth, box_bounds, num_points=10000, use_crop=False, use_fps=False):
        assert(depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1])

        # print(f"{far=}")
    
        # Create meshgrid for pixel coordinates
        xmap = np.arange(color.shape[1])
        ymap = np.arange(color.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        # Calculate 3D coordinates
        points_z = depth / self.camera_info.scale
        points_x = (xmap - self.camera_info.cx) * points_z / self.camera_info.fx
        points_y = (ymap - self.camera_info.cy) * points_z / self.camera_info.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1).astype(np.float32)
        cloud = cloud.reshape([-1, 3])
        
        # Clip points based on depth, right is x+
        mask = (cloud[:, 0] > box_bounds[0]) & (cloud[:, 0] < box_bounds[1]) \
             & (cloud[:, 1] > box_bounds[2]) & (cloud[:, 1] < box_bounds[3]) \
             & (cloud[:, 2] > box_bounds[4]) & (cloud[:, 2] < box_bounds[5])
        cloud = cloud[mask]

        # print("shape 0:", cloud.shape)
        if self.use_grid_sampling:
            cloud = grid_sample_pcd(cloud, grid_size=0.003)
        # print("shape 1:", cloud.shape)
        
        if use_crop:
            cloud = self.crop_point_cloud(cloud)
        if num_points > cloud.shape[0]:
            num_pad = num_points - cloud.shape[0]
            pad_points = np.zeros((num_pad, 3), dtype=np.float32)
            cloud = np.concatenate([cloud, pad_points], axis=0)
        elif use_fps:
            cloud, _ = farthest_point_sampling(cloud, num_points, use_cuda=True)
        else: 
            # Randomly sample points
            selected_idx = np.random.choice(cloud.shape[0], num_points, replace=True)
            cloud = cloud[selected_idx]
        
        # shuffle
        np.random.shuffle(cloud)
        # print("shape 2:", cloud.shape)
        return cloud.astype(np.float32)
    

def farthest_point_sampling(points, num_points=1024, use_cuda=True, use_numpy=False):
    # if use_numpy:
    #     sampled_points, indices = sample_farthest_points(points=points, K=num_points)
    #     return sampled_points, indices

    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices.cpu()


def sample_farthest_points(points, K, random_start_point=True):
    """
    最远点采样实现 (NumPy版本)
    
    参数:
        points: 输入点云，形状为 [B, N, 3] 或 [1, N, 3]
        K: 需要采样的点数
        random_start_point: 是否随机选择起始点
    
    返回:
        selected_points: 采样得到的点，形状为 [B, K, 3]
        indices: 采样点的索引，形状为 [B, K]
    """
    # 确保输入是3维的 [B, N, 3]
    if points.ndim == 2:
        points = np.expand_dims(points, 0)
    
    B, N, _ = points.shape
    device = points.device if hasattr(points, 'device') else None
    
    # 初始化输出
    selected_points = np.zeros((B, K, 3), dtype=points.dtype)
    indices = np.zeros((B, K), dtype=np.int64)
    
    for b in range(B):
        # 当前批次的点云
        pc = points[b]
        
        # 初始化距离数组，记录每个点到已选点集的最小距离
        min_distances = np.full(N, np.inf)
        
        # 选择第一个点
        if random_start_point:
            start_idx = np.random.randint(N)
        else:
            start_idx = 0
        selected_indices = [start_idx]
        
        # 计算初始距离
        selected_point = pc[start_idx]
        diff = pc - selected_point
        curr_distances = np.sum(diff ** 2, axis=1)
        min_distances = np.minimum(min_distances, curr_distances)
        
        for i in range(1, K):
            # 找到距离最远的点
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(farthest_idx)
            
            # 更新距离
            selected_point = pc[farthest_idx]
            diff = pc - selected_point
            curr_distances = np.sum(diff ** 2, axis=1)
            min_distances = np.minimum(min_distances, curr_distances)
        
        # 保存结果
        selected_points[b] = pc[selected_indices]
        indices[b] = np.array(selected_indices)
    
    return selected_points, indices


class PointCloudGenerator(Camera):
    def __init__(self, config: RealsensePointCloudCameraConfig):
        
        self.config = config
        self.serial_number = config.serial_number

        self.queue = Queue(maxsize=3)

        self.color_image = None
        self.depth_map = None
        self.points = None
        self.thread = None
        self.stop_event = None
        self.fps = 30
        self.logs = {}
        self.box_bounds = config.box_bounds
        self.height, self.width, self.channels = 540, 960, 3

        # sync_mode: Use 1 for master, 2 for slave, 0 for default (no sync)
        self.process = SingleVisionProcess(config.device_name, config.serial_number, self.queue,
                        enable_rgb=True, enable_depth=True, enable_pointcloud=True, sync_mode=1,
                        num_points=config.num_points, box_bounds=self.box_bounds,
                        use_grid_sampling=config.use_grid_sampling, use_crop=config.use_crop, img_size=config.img_size)

        self.process.start()
        print(f"PointCloudGenerator-{config.device_name}-{config.serial_number} start.")

    def __call__(self):  
        color, depth, point_cloud = self.queue.get()
        return color, depth, point_cloud

    def finalize(self):
        self.process.terminate()
    
    def connect(self):
        pass

    def read_loop(self):
        while not self.stop_event.is_set():
            self.color_image, self.depth_map, self.points = self.read()

    def read(self):
        start_time = time.perf_counter()

        cam_dict = self()

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        # print(f"Get takes {(time.perf_counter() - start_time) * 1000}")
        return cam_dict

    def async_read(self):
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )
        return self.color_image, self.depth_map, self.points
    
    def disconnect(self):
        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None
        self.finalize()

    def __del__(self):
        self.disconnect()
        

# if __name__ == "__main__":
#     cam = MultiRealSense(use_right_cam=False, front_num_points=20000, 
#                          use_grid_sampling=True, use_crop=False, img_size=1024)
#     import matplotlib.pyplot as plt
#     while True:
#         out = cam()
#         print(out.keys())
    
#         imageio.imwrite(f'color_front.png', out['color'])
#         print("save to color_front.png")
#         # imageio.imwrite(f'color_right.png', out['right_color'])
#         # imageio.imwrite(f'depth_right.png', out['right_depth'])
#         # imageio.imwrite(f'depth_front.png', out['right_front'])
#         plt.imshow(out['depth'])
#         plt.savefig("depth_front.png")
#         print("save to depth_front.png")
#         import visualizer
#         # visualizer.visualize_pointcloud(out['right_point_cloud'])
#         visualizer.visualize_pointcloud(out['front_point_cloud'])
#         cam.finalize()