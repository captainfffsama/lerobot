# Meta Quest 3 Teleoperator

This teleoperator provides integration with Meta Quest 3 VR controllers for robot teleoperation using lerobot.

## Features

- **6DOF Control**: Full 6-degree-of-freedom control using both VR controllers
- **Dual Hand Support**: Independent control for left and right hands
- **Button/Trigger Inputs**: Access to all controller buttons and triggers
- **Real-time Tracking**: Low-latency pose tracking from Meta Quest 3
- **Calibration Support**: Automatic calibration for accurate control

## Action Features

The teleoperator provides the following action features:

### Left Hand
- `left_hand.pos`: 3D position (numpy array, shape (3,))
- `left_hand.rot`: Quaternion rotation (numpy array, shape (4,))
- `left_hand.grip`: Grip value [0, 1] (float)
- `left_hand.trigger`: Trigger value [0, 1] (float)
- `left_hand.buttons`: Button states (dict)

### Right Hand
- `right_hand.pos`: 3D position (numpy array, shape (3,))
- `right_hand.rot`: Quaternion rotation (numpy array, shape (4,))
- `right_hand.grip`: Grip value [0, 1] (float)
- `right_hand.trigger`: Trigger value [0, 1] (float)
- `right_hand.buttons`: Button states (dict)

### Control
- `enabled`: Whether teleoperation is active (bool)

## Configuration

```python
from lerobot.teleoperators.meta_quest3.config_meta_quest3 import MetaQuest3Config

config = MetaQuest3Config(
    id="meta_quest3_demo",
    ipaddress="192.168.1.100",  # Meta Quest 3 IP address
    port="30001"                # TCP port for communication
)
```

## Usage

### Basic Usage

```python
from lerobot.teleoperators.meta_quest3.teleop_meta_quest3 import MetaQuest3Teleop

# Create teleoperator
teleop = MetaQuest3Teleop(config)

# Connect and calibrate
teleop.connect(calibrate=True)

# Get current action
action = teleop.get_action()
print(f"Left hand position: {action['left_hand.pos']}")
print(f"Right hand position: {action['right_hand.pos']}")
print(f"Teleoperation enabled: {action['enabled']}")

# Disconnect
teleop.disconnect()
```

### Recording Demonstrations

```python
from lerobot.record import record

# Record demonstrations
record(
    teleop=teleop,
    output_dir="./recordings",
    episode_length=1000
)
```

## Setup Requirements

1. **Meta Quest 3 Device**: Physical Meta Quest 3 headset with controllers
2. **Network Connection**: Meta Quest 3 and computer on same network
3. **Meta Quest 3 App**: Custom app running on Meta Quest 3 to send tracking data
4. **TCP Server**: The teleoperator runs a TCP server to receive data

## Network Protocol

The teleoperator communicates with Meta Quest 3 using a custom TCP protocol:

- **Port**: Configurable (default: 30001)
- **Protocol**: Binary packets with JSON payload
- **Data Types**: 
  - Heartbeat packets (0x23)
  - Controller function packets (0x6D)
- **Tracking Data**: JSON format with pose and input information

## Calibration

The teleoperator supports automatic calibration:

1. Hold both controllers in a neutral position
2. Press and hold the grip buttons on both controllers
3. The system will capture the reference poses
4. All subsequent movements are relative to these reference poses

## Button Mapping

### Available Buttons
- `primaryButton`: Primary button (A/X)
- `secondaryButton`: Secondary button (B/Y)
- `menuButton`: Menu button
- `axisClick`: Joystick click

### Input Values
- `grip`: Grip button pressure [0, 1]
- `trigger`: Trigger pressure [0, 1]
- `axisX`, `axisY`: Joystick axes [-1, 1]

## Troubleshooting

### Connection Issues
- Ensure Meta Quest 3 and computer are on the same network
- Check IP address and port configuration
- Verify Meta Quest 3 app is running and sending data

### No Tracking Data
- Check if Meta Quest 3 controllers are powered on
- Ensure controllers are being tracked by the headset
- Verify the tracking app is running on Meta Quest 3

### Calibration Issues
- Make sure both controllers are visible to the headset
- Hold controllers in a stable position during calibration
- Try recalibrating if movements seem incorrect

## Example Files

- `test_meta_quest3.py`: Basic functionality test
- `examples/meta_quest3/record.py`: Recording example
- `meta_quest3_server.py`: TCP server implementation
- `decoder.py`: Network protocol decoder

## Integration with lerobot

The Meta Quest 3 teleoperator is fully compatible with lerobot's recording and replay functionality:

```python
# Recording
from lerobot.record import record
record(teleop=teleop, output_dir="./data")

# Replay
from lerobot.replay import replay
replay(dataset_path="./data")
```

## Technical Details

- **Threading**: Server runs in separate thread for non-blocking operation
- **Data Buffering**: Handles packet concatenation and partial data
- **Pose Tracking**: 6DOF pose tracking with quaternion rotations
- **Input Processing**: Real-time button and trigger state monitoring
- **Calibration**: Position and orientation reference frame setup
