import json

import pytest

from pylibxdaq.device import DeviceInfo, scan_devices, list_devices
from pylibxdaq import pyxdaq_device

from .test_lifetime_manager import mock_manager_path


@pytest.fixture()
def mock_paths(mock_manager_path):
    return [mock_manager_path]


@pytest.fixture()
def mock_dir(mock_manager_path):
    return mock_manager_path.parent


def test_scan_devices_returns_raw_results(mock_paths):
    results = scan_devices(mock_paths)
    assert len(results) > 0
    assert all(
        isinstance(
            r, (
                pyxdaq_device.DeviceFound, pyxdaq_device.DeviceOccupied,
                pyxdaq_device.DeviceQueryFailed
            )
        ) for r in results
    )


def test_list_devices_returns_device_infos(mock_dir):
    devices = list_devices(mock_dir)
    assert len(devices) == 3
    assert all(isinstance(d, DeviceInfo) for d in devices)


def test_list_devices_metadata_populated(mock_dir):
    devices = list_devices(mock_dir)
    for d in devices:
        assert d.serial_number.startswith("MOCK")
        assert d.api_version == "mock-1.0"
        assert isinstance(d.info, dict)
        assert isinstance(d.status, dict)
        assert "Serial Number" in d.info
        assert "API" in d.status
        assert isinstance(d.manager_info, dict)
        assert d.manager_info["name"] == "Mock Device Manager"


def test_list_devices_options_contain_id(mock_dir):
    devices = list_devices(mock_dir)
    ids = {d.options["id"] for d in devices}
    assert ids == {0, 1, 2}


def test_list_devices_sorted(mock_dir):
    devices = list_devices(mock_dir)
    keys = [str(d.options) for d in devices]
    assert keys == sorted(keys)


def test_list_devices_filters_occupied(mock_dir, mock_manager_path):
    # Hold one device open — list_devices should not include it.
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    try:
        devices = list_devices(mock_dir)
        ids = {d.options["id"] for d in devices}
        assert 0 not in ids
        assert ids == {1, 2}
    finally:
        device.close()


def test_list_devices_defaults_to_manager_dir(monkeypatch, mock_dir):
    import pylibxdaq.device as dev_module
    monkeypatch.setattr(dev_module, "DeviceManagerDir", mock_dir)
    devices = list_devices()
    assert len(devices) == 3


# ---------------------------------------------------------------------------
# DeviceInfo
# ---------------------------------------------------------------------------


def test_with_mode_returns_copy(mock_dir):
    d = list_devices(mock_dir)[0]
    d2 = d.with_mode("rhd")
    assert d2.options["mode"] == "rhd"
    assert "mode" not in d.options  # original unmodified


def test_create_returns_device(mock_dir):
    info = list_devices(mock_dir)[0]
    with info.create() as dev:
        assert dev.raw is not None
        assert not dev.raw.closed


def test_create_device_info_attached(mock_dir):
    info = list_devices(mock_dir)[0]
    with info.create() as dev:
        assert dev.device_info is info
        assert isinstance(dev.status, dict)
        assert isinstance(dev.info, dict)


def test_device_closed_after_context_exit(mock_dir):
    info = list_devices(mock_dir)[0]
    with info.create() as dev:
        raw = dev.raw
    assert dev.raw is None
    assert raw.closed
