import json
from pathlib import Path

import pytest

from pylibxdaq import pyxdaq_device
from pylibxdaq.managers import DeviceManagerDir

# ---------------------------------------------------------------------------
# Known managers — add new entries here to extend coverage
# ---------------------------------------------------------------------------

KNOWN_MANAGERS = [
    "thor_device_manager",
    "ok_device_manager",
]


def _find_manager(name_fragment: str) -> Path:
    """Return the deployed manager .so/.dll/.dylib matching name_fragment."""
    candidates = [
        p for p in DeviceManagerDir.iterdir()
        if p.is_file() and name_fragment in p.name and p.suffix in (".so", ".dll", ".dylib")
    ]
    assert candidates, (
        f"No deployed manager matching '{name_fragment}' found in {DeviceManagerDir}. "
        "Run ./build.sh first."
    )
    return candidates[0]


@pytest.fixture(scope="module")
def manager_paths() -> dict[str, Path]:
    return {name: _find_manager(name) for name in KNOWN_MANAGERS}


# ---------------------------------------------------------------------------
# Layout checks
# ---------------------------------------------------------------------------


def test_managers_dir_exists():
    assert DeviceManagerDir.is_dir(), f"Managers directory not found: {DeviceManagerDir}"


def test_libxdaq_device_deployed():
    """xdaq_device shared lib must be co-located with pyxdaq_device for RPATH/DLL lookup."""
    package_root = Path(pyxdaq_device.__file__).parent
    import sys
    if sys.platform == "win32":
        libs = list(package_root.glob("xdaq_device.dll"))
    else:
        libs = list(package_root.glob("libxdaq_device*"))
    assert libs, (
        f"xdaq_device shared library not found in {package_root}. "
        "Check xdaq_install_core() in CMakeLists.txt."
    )


def test_resources_deployed():
    resources_dir = DeviceManagerDir.parent / "resources"
    assert resources_dir.is_dir(), f"resources/ directory not found at {resources_dir}"
    bitfiles = list(resources_dir.glob("*.bit"))
    assert bitfiles, f"No .bit files found in {resources_dir}"


def test_all_known_managers_present():
    names = {p.name for p in DeviceManagerDir.iterdir() if p.is_file()}
    for fragment in KNOWN_MANAGERS:
        assert any(fragment in n for n in names), f"{fragment} not deployed"


# ---------------------------------------------------------------------------
# Per-manager plugin loading — parametrized over KNOWN_MANAGERS
# ---------------------------------------------------------------------------


@pytest.fixture(params=KNOWN_MANAGERS)
def manager_path(request, manager_paths):
    return manager_paths[request.param]


def test_manager_loads(manager_path):
    assert pyxdaq_device.get_device_manager(manager_path) is not None


def test_manager_info(manager_path):
    manager = pyxdaq_device.get_device_manager(manager_path)
    info = json.loads(manager.info())
    assert isinstance(info, dict), "info() must return valid JSON object"


def test_manager_list_devices_is_json(manager_path):
    manager = pyxdaq_device.get_device_manager(manager_path)
    devices = json.loads(manager.list_devices())
    assert isinstance(devices, list), "list_devices() must return a JSON array"


def test_manager_get_device_options_is_json(manager_path):
    manager = pyxdaq_device.get_device_manager(manager_path)
    opts = json.loads(manager.get_device_options())
    assert isinstance(opts, dict), "get_device_options() must return a JSON object"


def test_manager_location(manager_path):
    manager = pyxdaq_device.get_device_manager(manager_path)
    loc = manager.location()
    assert Path(loc).exists(), f"manager.location() returned non-existent path: {loc}"


# ---------------------------------------------------------------------------
# scan_devices_dir — round-trip through deployed managers
# ---------------------------------------------------------------------------


def test_scan_devices_dir_uses_deployed_managers():
    """scan_devices_dir should not raise even when no hardware is attached."""
    results = pyxdaq_device.scan_devices_dir(DeviceManagerDir)
    assert isinstance(results, list)


def test_scan_devices_dir_no_unknown_types():
    results = pyxdaq_device.scan_devices_dir(DeviceManagerDir)
    for r in results:
        assert isinstance(
            r,
            (
                pyxdaq_device.DeviceFound,
                pyxdaq_device.DeviceOccupied,
                pyxdaq_device.DeviceQueryFailed,
            ),
        )
