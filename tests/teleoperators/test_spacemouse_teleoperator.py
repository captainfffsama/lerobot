import types
from unittest.mock import patch
import pytest

from lerobot.teleoperators.spacemouse import (
    SpacemouseTeleoperator,
    SpacemouseTeleoperatorConfig,
)


class DummyState:
    def __init__(self, x=0.1, y=-0.2, z=0.05, roll=0.01, pitch=-0.02, yaw=0.03, buttons=(0.0, 0.0), t=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.buttons = buttons
        self.t = t


@pytest.fixture
def teleop():
    cfg = SpacemouseTeleoperatorConfig(
        id="test_spacemouse",
        scale_translation=2.0,
        scale_rotation=10.0,
        deadzone=0.0,
    )
    with patch("lerobot.teleoperators.spacemouse.spacemouse_teleoperator.pyspacemouse") as m:
        m.open.return_value = True
        m.read.return_value = DummyState()
        t = SpacemouseTeleoperator(cfg)
        yield t, m
        t.disconnect()


def test_connect_disconnect(teleop):
    t, m = teleop
    assert not t.is_connected
    t.connect()
    assert t.is_connected
    t.disconnect()
    assert not t.is_connected
    m.close.assert_called()


def test_get_action(teleop):
    t, m = teleop
    t.connect()
    m.read.return_value = DummyState()
    act = t.get_action()
    for k in ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw", "gripper"]:
        assert k in act
    # scale 应生效
    assert act["delta_x"] == pytest.approx(0.1 * t.config.scale_translation)
    assert act["delta_roll"] == pytest.approx(0.01 * t.config.scale_rotation)


def test_not_connected_error(teleop):
    t, _ = teleop
    with pytest.raises(RuntimeError):
        t.get_action()


def test_spacemouse_act():
    config = SpacemouseTeleoperatorConfig()
    teleop = SpacemouseTeleoperator(config)

    teleop.connect()
    import time

    while True:
        try:
            time.sleep(0.5)

            action = teleop.get_action()
            print(action)
        except RuntimeError:
            print("spacemouse not connected")

test_spacemouse_act()
