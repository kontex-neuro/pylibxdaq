import json

from pylibxdaq import pyxdaq_device

from .test_lifetime_manager import mock_manager_path


def test_scan_devices_found(mock_manager_path):
    results = pyxdaq_device.scan_devices([mock_manager_path])

    found = [r for r in results if isinstance(r, pyxdaq_device.DeviceFound)]
    assert len(found) == 3  # mock manager exposes ids 0, 1, 2

    serials = {r.serial_number for r in found}
    assert serials == {"MOCK000", "MOCK001", "MOCK002"}

    for r in found:
        assert r.api_version == "mock-1.0"
        assert r.device_manager_path == mock_manager_path

        config = json.loads(r.device_config_json)
        assert "id" in config

        info = json.loads(r.info_json)
        assert "Serial Number" in info

        status = json.loads(r.status_json)
        assert "API" in status


def test_scan_devices_repr(mock_manager_path):
    results = pyxdaq_device.scan_devices([mock_manager_path])
    for r in results:
        assert repr(r)  # should not raise or return empty string


def test_scan_devices_empty_path_list():
    results = pyxdaq_device.scan_devices([])
    assert results == []


def test_scan_devices_nonexistent_path():
    # Managers that fail to load are silently skipped.
    results = pyxdaq_device.scan_devices(["/nonexistent/path/device_manager.so"])
    assert results == []


def test_scan_devices_dir(mock_manager_path):
    results = pyxdaq_device.scan_devices_dir(mock_manager_path.parent)

    found = [r for r in results if isinstance(r, pyxdaq_device.DeviceFound)]
    assert len(found) >= 3  # at minimum the 3 mock devices


def test_scan_devices_dir_nonexistent():
    results = pyxdaq_device.scan_devices_dir("/nonexistent/directory")
    assert results == []


def test_scan_occupied_device(mock_manager_path):
    # Hold a device open in this process, then scan — it should appear as occupied.
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    results = pyxdaq_device.scan_devices([mock_manager_path])

    occupied = [r for r in results if isinstance(r, pyxdaq_device.DeviceOccupied)]
    found = [r for r in results if isinstance(r, pyxdaq_device.DeviceFound)]

    assert len(occupied) >= 1
    assert len(found) == 2  # ids 1 and 2 remain free

    device.close()
