# PySpaceMouse Setup Guide for Ubuntu

This guide provides instructions for setting up the [PySpaceMouse library](https://github.com/JakubAndrysek/PySpaceMouse) on an Ubuntu system to use a 3Dconnexion SpaceMouse device.

## Environment

  * **OS:** Ubuntu 24.04.3
  * **Python:** 3.10
  * **Library:** `pyspacemouse`

## Installation Steps

Follow these steps to install the required dependencies, configure device permissions, and install the Python library.

### 1. Install System Dependencies

First, you need to install the `libhidapi` development library, which allows Python to communicate with raw HID devices.

```bash
sudo apt-get update
sudo apt-get install libhidapi-dev
```

### 2. Configure Device Permissions

By default, Linux requires root privileges to access raw HID devices. To allow access for your user, create a `udev` rule that grants read/write permissions to members of the `plugdev` group.

Execute the following command to create the rule file:

```bash
echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-hidraw-permissions.rules
```

*(Note: This command creates the permissions rule. For the system to recognize it immediately, you could run `sudo udevadm control --reload-rules && sudo udevadm trigger`, but a reboot will also apply the rule.)*

### 3. Add User to the `plugdev` Group

Next, add your current user account to the `plugdev` group to inherit the permissions defined in the `udev` rule.

```bash
sudo usermod -aG plugdev $USER
```

### 4. Apply New Group Membership

For the group change to take effect in your current terminal session, run the `newgrp` command. **Alternatively, you can simply log out and log back in.**

```bash
newgrp plugdev
```

You might be prompted to enter your password. This command starts a new shell with the updated group permissions.

### 5. Install Python Packages

Finally, use `pip` to install `easyhid` and the `pyspacemouse` library.

```bash
pip install easyhid
pip install pyspacemouse
```

## Verification

Your setup is now complete. To test if the device is correctly recognized, you can connect your SpaceMouse and run one of the example scripts from the [PySpaceMouse GitHub repository](https://www.google.com/search?q=https://github.com/JakubAndrysek/PySpaceMouse/tree/master/examples).