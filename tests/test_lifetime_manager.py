import gc
import json
import pytest
from pylibxdaq import pyxdaq_device
from pathlib import Path


class PrintWhenExitProcess:
    allmessages = []

    def __del__(self):
        print("Captured messages during tests:")
        for msg in self.allmessages:
            print(f"{msg.out.strip()}")
            print(f"{msg.err.strip()}")


# print_on_exit = PrintWhenExitProcess()
#
#
# @pytest.fixture()
# def capfd(capfd):
#
#     class CapFDWrapper:
#
#         def __init__(self, capfd):
#             self._capfd = capfd
#
#         def readouterr(self):
#             captured = self._capfd.readouterr()
#             PrintWhenExitProcess.allmessages.append(captured)
#             return captured
#
#     return CapFDWrapper(capfd)


@pytest.fixture(scope="module")
def mock_manager_path():
    manager_path = Path(__file__).parent.parent / "build"
    mock_devices = [
        p for p in manager_path.rglob("**/mock_device_manager*")
        if p.suffix in ['.so', '.dylib', '.dll']
    ]
    assert len(mock_devices) > 0
    mock_devices.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    manager_path = mock_devices[0]
    assert manager_path.exists()
    return manager_path


def test_cpython_immediate_dealloc(mock_manager_path, capfd):
    """Test CPython's immediate deallocation via reference counting (without gc)"""
    gc.disable()
    try:
        manager = pyxdaq_device.get_device_manager(mock_manager_path)

        info = json.loads(manager.info())
        assert info["name"] == "Mock Device Manager"

        captured = capfd.readouterr()
        assert "MockDeviceManager constructed" in captured.err
        assert "MockDeviceManager destroyed" not in captured.err
        del manager
        captured = capfd.readouterr()
        assert "MockDeviceManager destroyed" in captured.err
    finally:
        gc.enable()


def test_manager_caching(mock_manager_path, capfd):
    manager1 = pyxdaq_device.get_device_manager(mock_manager_path)

    captured = capfd.readouterr()
    assert captured.err.count("MockDeviceManager constructed") == 1
    assert "total instances: 1" in captured.err

    manager2 = pyxdaq_device.get_device_manager(mock_manager_path)
    manager3 = pyxdaq_device.get_device_manager(mock_manager_path)

    captured = capfd.readouterr()
    assert captured.err.count("MockDeviceManager constructed") == 0

    assert manager1.location() == manager2.location() == manager3.location()

    del manager1

    captured = capfd.readouterr()
    assert captured.err.count("MockDeviceManager destroyed") == 0

    assert manager2.list_devices() is not None
    assert manager3.list_devices() is not None

    del manager2

    captured = capfd.readouterr()
    assert captured.err.count("MockDeviceManager destroyed") == 0

    assert manager3.list_devices() is not None

    del manager3

    captured = capfd.readouterr()
    assert captured.err.count("MockDeviceManager destroyed") == 1


def test_manager_reload(mock_manager_path, capfd):
    manager1 = pyxdaq_device.get_device_manager(mock_manager_path)
    location1 = manager1.location()

    captured = capfd.readouterr()
    assert "MockDeviceManager constructed" in captured.err
    assert "MockDeviceManager destroyed" not in captured.err

    del manager1

    captured = capfd.readouterr()
    assert "MockDeviceManager destroyed" in captured.err

    manager2 = pyxdaq_device.get_device_manager(mock_manager_path)
    location2 = manager2.location()
    assert location1 == location2

    captured = capfd.readouterr()
    assert "MockDeviceManager constructed" in captured.err
    assert "MockDeviceManager destroyed" not in captured.err

    del manager2

    captured = capfd.readouterr()
    assert "MockDeviceManager destroyed" in captured.err
