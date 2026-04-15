import gc
import json

import pytest

from pylibxdaq import pyxdaq_device

from .test_lifetime_manager import mock_manager_path


def test_device_survives_manager_deletion(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    devices = json.loads(manager.list_devices())
    assert len(devices) > 0

    captured = capfd.readouterr()
    assert "MockDeviceManager constructed" in captured.err
    assert "MockDeviceManager destroyed" not in captured.err

    device = manager.create_device(json.dumps(devices[0]))

    captured = capfd.readouterr()
    assert "MockDevice(0) constructed" in captured.err

    del manager

    captured = capfd.readouterr()
    assert "MockDeviceManager destroyed" not in captured.err
    assert "MockDevice(0) destroyed" not in captured.err
    assert device.get_info() is not None

    del device

    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDeviceManager destroyed" in captured.err


def test_cpython_device_immediate_dealloc(mock_manager_path, capfd):
    """Test device immediate deallocation without gc"""
    gc.disable()
    try:
        manager = pyxdaq_device.get_device_manager(mock_manager_path)
        devices = json.loads(manager.list_devices())

        capfd.readouterr()

        device = manager.create_device(json.dumps(devices[0]))

        captured = capfd.readouterr()
        assert "MockDevice(0) constructed" in captured.err
        assert "MockDevice(0) destroyed" not in captured.err

        del manager

        captured = capfd.readouterr()
        assert "MockDeviceManager destroyed" not in captured.err
        assert "MockDevice(0) destroyed" not in captured.err

        del device

        captured = capfd.readouterr()
        assert "MockDevice(0) destroyed" in captured.err
        assert "MockDeviceManager destroyed" in captured.err
    finally:
        gc.enable()


def test_device_explicit_close(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))
    capfd.readouterr()

    device.close()
    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" in captured.err

    with pytest.raises(ValueError, match="Device is already closed"):
        device.read_registers()


def test_device_context_manager(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    with manager.create_device(json.dumps({"id": 0})) as device:
        capfd.readouterr()

    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" in captured.err

    with pytest.raises(ValueError, match="Device is already closed"):
        device.read_registers()


def test_multiple_devices_lifetime(mock_manager_path, capfd):

    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    captured = capfd.readouterr()
    assert "MockDeviceManager constructed" in captured.err
    assert "MockDeviceManager destroyed" not in captured.err

    device_list = json.loads(manager.list_devices())

    devices = []
    for dev_info in device_list[:3]:
        device_config = json.dumps({"id": dev_info['id']})
        devices.append(manager.create_device(device_config))

    captured = capfd.readouterr()
    assert "MockDevice(0) constructed" in captured.err
    assert "MockDevice(1) constructed" in captured.err
    assert "MockDevice(2) constructed" in captured.err
    assert "MockDevice(0) destroyed" not in captured.err

    devices.clear()

    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDevice(1) destroyed" in captured.err
    assert "MockDevice(2) destroyed" in captured.err

    del manager

    captured = capfd.readouterr()
    assert "MockDeviceManager destroyed" in captured.err
