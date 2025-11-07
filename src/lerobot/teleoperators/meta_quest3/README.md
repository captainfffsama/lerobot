# Meta Quest 3 遥操作器

本遥操作器为 lerobot 项目提供 Meta Quest 3 VR 控制器的集成，支持机器人遥操作和数据录制。

## 功能特性

- **6DOF 控制**：使用 VR 控制器进行完整的 6 自由度控制
- **双手支持**：支持左右手独立控制
- **实时跟踪**：低延迟的姿态跟踪和输入处理
- **增量控制**：输出 delta 动作，适合机器人控制
- **网络通信**：基于 TCP 的自定义协议通信

## 动作特征

遥操作器根据 `hand_name` 配置提供不同的动作特征（delta 格式）：

### 单手模式（hand_name = 'left' 或 'right'）
- `delta_x`：X 轴位置增量 (float)
- `delta_y`：Y 轴位置增量 (float)  
- `delta_z`：Z 轴位置增量 (float)
- `delta_roll`：绕 X 轴旋转增量 (float)
- `delta_pitch`：绕 Y 轴旋转增量 (float)
- `delta_yaw`：绕 Z 轴旋转增量 (float)

### 双手模式（hand_name = 'both'）
- `l_delta_x`, `l_delta_y`, `l_delta_z`：左手位置增量
- `l_delta_roll`, `l_delta_pitch`, `l_delta_yaw`：左手旋转增量
- `r_delta_x`, `r_delta_y`, `r_delta_z`：右手位置增量
- `r_delta_roll`, `r_delta_pitch`, `r_delta_yaw`：右手旋转增量

## 配置参数

```python
from lerobot.teleoperators.meta_quest3.config_meta_quest3 import MetaQuest3Config

config = MetaQuest3Config(
    id="meta_quest3_demo",
    ipaddress="192.168.1.100",  # Meta Quest 3 服务器 IP 地址
    port="30001",               # TCP 通信端口
    move_scale=1.0,             # 位置移动缩放因子
    rot_scale=1.0,              # 旋转缩放因子
    hand_name="left"            # 控制手：'left'、'right' 或 'both'
)
```

## 使用方法

### 基本使用

#### 单手模式
```python
from lerobot.teleoperators.meta_quest3.teleop_meta_quest3 import MetaQuest3Teleop

# 创建单手遥操作器
config = MetaQuest3Config(
    id="meta_quest3_left",
    ipaddress="192.168.1.100",
    port="30001",
    hand_name="left"  # 使用左手
)
teleop = MetaQuest3Teleop(config)

# 连接和校准
teleop.connect(calibrate=True)

# 获取当前动作
action = teleop.get_action()
print(f"X 轴增量: {action['delta_x']}")
print(f"Y 轴增量: {action['delta_y']}")
print(f"Z 轴增量: {action['delta_z']}")

# 断开连接
teleop.disconnect()
```

#### 双手模式
```python
# 创建双手遥操作器
config = MetaQuest3Config(
    id="meta_quest3_both",
    ipaddress="192.168.1.100",
    port="30001",
    hand_name="both"  # 使用双手
)
teleop = MetaQuest3Teleop(config)

# 连接和校准
teleop.connect(calibrate=True)

# 获取当前动作
action = teleop.get_action()
print(f"左手 X 轴增量: {action['l_delta_x']}")
print(f"右手 X 轴增量: {action['r_delta_x']}")

# 断开连接
teleop.disconnect()
```

### 录制演示数据

```python
from lerobot.record import record

# 录制机器人演示
record(
    teleop=teleop,
    output_dir="./recordings",
    episode_length=1000
)
```

## 文件结构

```
src/lerobot/teleoperators/meta_quest3/
├── __init__.py                    # 模块初始化
├── config_meta_quest3.py         # 配置类定义
├── teleop_meta_quest3.py         # 主要遥操作器类
├── meta_quest3_server.py         # TCP 服务器实现
├── net_package_handler.py        # 网络数据包处理器
└── README.md                     # 本文档

examples/meta_quest3_to_ur/
├── record.py                     # 录制示例（Meta Quest 3 → UR5）
└── replay.py                     # 回放示例
```

## 核心组件

### 1. MetaQuest3Teleop（遥操作器类）
- 继承自 `Teleoperator` 基类
- 实现所有必需的抽象方法
- 提供 delta 格式的动作输出

### 2. MetaQuest3Server（TCP 服务器）
- 处理来自 Meta Quest 3 的网络数据
- 支持数据包缓冲和解析
- 线程安全的实时数据传输

### 3. NetPackageHandler（数据包处理器）
- 解析自定义网络协议
- 支持心跳包和控制器数据包
- 处理 JSON 格式的跟踪数据

## 网络协议

### 数据包格式
```
[头部(1字节)] + [命令(1字节)] + [长度(4字节)] + [数据(n字节)] + [时间戳(8字节)] + [尾部(1字节)]
```

### 支持的命令
- `0x23`：客户端心跳包
- `0x6D`：控制器功能数据包

### 跟踪数据格式
```json
{
  "functionName": "Tracking",
  "data": {
    "Controller": {
      "left": {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "grip": 0.0,
        "trigger": 0.0,
        "primaryButton": false,
        "secondaryButton": false,
        "menuButton": false,
        "axisClick": false
      },
      "right": {
        // 同样的数据结构
      }
    }
  }
}
```

## 设置要求

### 硬件要求
1. **Meta Quest 3 设备**：物理 VR 头显和控制器
2. **网络连接**：Meta Quest 3 和计算机在同一网络
3. **Meta Quest 3 应用**：运行自定义应用发送跟踪数据

### 软件要求
1. **Python 依赖**：
   - `numpy`
   - `scipy`
   - `opencv-python`
   - `lerobot`

2. **网络配置**：
   - 确保端口未被占用
   - 配置防火墙允许连接

## 控制说明

### 启用遥操作
- 按住指定手的 **扳机键** 启用控制
- 松开扳机键停止控制（输出零增量）

### 移动控制
- **位置**：移动控制器产生位置增量
- **旋转**：旋转控制器产生旋转增量
- **缩放**：通过 `move_scale` 和 `rot_scale` 参数调整

### 按钮映射
- `primaryButton`：主按钮（A/X）
- `secondaryButton`：次按钮（B/Y）
- `menuButton`：菜单按钮
- `axisClick`：摇杆点击
- `grip`：握持按钮
- `trigger`：扳机按钮

## 故障排除

### 连接问题
- 检查 IP 地址和端口配置
- 确认 Meta Quest 3 应用正在运行
- 验证网络连接状态

### 数据问题
- 确认控制器被头显正确跟踪
- 检查跟踪应用是否发送数据
- 查看服务器日志输出

### 控制问题
- 调整 `move_scale` 和 `rot_scale` 参数
- 确认扳机键正常工作
- 检查动作特征配置

## 示例代码

### 录制 UR5 机器人演示
```python
# 运行录制示例
python examples/meta_quest3_to_ur/record.py
```

### 回放录制的数据
```python
# 运行回放示例
python examples/meta_quest3_to_ur/replay.py
```