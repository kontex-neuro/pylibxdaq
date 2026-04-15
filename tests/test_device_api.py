import json
import numpy as np
import pytest
from pylibxdaq import pyxdaq_device
from .test_lifetime_manager import mock_manager_path


def test_register_methods(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    assert device.set_register_sync(0x00, 0xFF) == pyxdaq_device.ReturnCode.Success
    assert device.set_register_sync(0x00, 0xFF, 0xFF) == pyxdaq_device.ReturnCode.Success
    assert device.set_register(0x00, 0xFF) == pyxdaq_device.ReturnCode.Success
    assert device.set_register(0x00, 0xFF, 0xFF) == pyxdaq_device.ReturnCode.Success
    assert device.send_registers() == pyxdaq_device.ReturnCode.Success
    assert isinstance(device.get_register(0x00), int)
    device.get_register_sync(0x00)  # returns int or None
    assert device.read_registers() == pyxdaq_device.ReturnCode.Success
    assert device.trigger(0x00, 0) == pyxdaq_device.ReturnCode.Success


def test_raw_io(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    data = np.zeros(16, dtype=np.uint8)
    assert device.write(0x00, data) == 16

    buf = np.zeros(16, dtype=np.uint8)
    assert device.read(0x00, buf) == 16


def test_get_status(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    status = device.get_status()
    assert isinstance(status, str)
    json.loads(status)  # must be valid JSON


def test_get_info(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    info = device.get_info()
    assert isinstance(info, str)
    json.loads(info)  # must be valid JSON


def test_manager_get_device_options(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    options = json.loads(manager.get_device_options())
    assert "properties" in options


def test_stream_raises_after_context_exit(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    def cb(event, err):
        pass

    with device.start_read_stream(0x11, cb) as stream:
        pass

    with pytest.raises(ValueError):
        with stream:
            pass
