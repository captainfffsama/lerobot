# Setup Guide for `joycon-python` on Ubuntu

This guide provides instructions for setting up the [`joycon-python`](https://www.google.com/search?q=%5Bhttps://github.com/captainfffsama/joycon-python%5D\(https://github.com/captainfffsama/joycon-python\)) library on an Ubuntu system to use Nintendo Switch Joy-Cons and Pro Controllers.

## Environment

  * **OS:** Ubuntu 24.04.3
  * **Python:** 3.10

## Setup Instructions

### 1\. Install Dependencies

First, install the required system libraries and Python packages. `libhidapi-dev` is necessary for direct communication with HID devices.

```bash
# Update package list
sudo apt update

# Install the HID API library
sudo apt install libhidapi-dev

# Install required Python packages
pip install hid hidapi pyglm joycon-python
```

### 2\. Configure udev Rules for Device Permissions

By default, Linux may mount Nintendo Switch controllers as read-only devices, which prevents applications from sending the initialization commands needed to use them properly.

To fix this, we will create a `udev` rule to grant write permissions for these devices.

Create a new file at `/etc/udev/rules.d/50-nintendo-switch.rules` with the following content. You can do this easily with the command below:

```bash
sudo tee /etc/udev/rules.d/50-nintendo-switch.rules > /dev/null <<EOF
# Rule for Nintendo Switch Controllers
# Grants write permissions to the hidraw device, allowing user-space drivers to initialize and use the controllers.
#
# Reference: https://www.reddit.com/r/Stadia/comments/egcvpq/comment/fc5s7qm/

# Switch Joy-con (L) (Bluetooth only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2006.*", MODE="0666"

# Switch Joy-con (R) (Bluetooth only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2007.*", MODE="0666"

# Switch Pro controller (USB and Bluetooth)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009", MODE="0666"
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2009.*", MODE="0666"

# Switch Joy-con charging grip (USB only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="200e", MODE="0666"
EOF
```

After creating the rule file, you must reload the `udev` rules and re-plug your device for the changes to take effect.

```bash
# Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Now, disconnect and reconnect your controller (or turn it off and on if connected via Bluetooth).

### 3\. Test the Setup

You can verify that the library can connect to your controller using the following Python script. This example attempts to connect to a right Joy-Con.

```python
# test_joycon.py
from pyjoycon import JoyCon, get_R_id
import logging

# Optional: Enable logging to see detailed output
# logging.basicConfig(level=logging.DEBUG)

try:
    # Find the device ID for the right Joy-Con
    joycon_id = get_R_id()
    print(f"Found Right Joy-Con with ID: {joycon_id}")

    # Initialize the JoyCon object
    joycon = JoyCon(*joycon_id)

    # Get and print the device status
    status = joycon.get_status()
    print("Successfully connected. Device status:")
    print(status)

except ValueError:
    print("Right Joy-Con not found. Please make sure it is connected.")

```

Save the code as `test_joycon.py` and run it:

```bash
python test_joycon.py
```

If the connection is successful, you will see the device ID and its status printed to the console without any permission errors.