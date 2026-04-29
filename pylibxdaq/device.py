"""
pylibxdaq: Python binding layer for XDAQ device management

This module provides a high-level interface to enumerate and access
XDAQ devices using dynamically loaded device managers.

Note:
    Devices must be used as context managers to ensure proper resource release.
    Re-opening a device without cleanup may lead to hardware conflicts.
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Self, Union

from . import pyxdaq_device
from .managers import DeviceManagerDir

__all__ = ["DeviceInfo", "Device", "list_devices", "scan_devices"]


@dataclass
class Device:
    """
    Represents a concrete XDAQ device created via a backend device manager.

    This class wraps a low-level device binding (`raw`) along with parsed metadata
    from the manager. Must be used as a context manager to ensure hardware is released.

    Example:
        with list_devices()[0].with_mode('rhd').create() as dev:
            dev.raw.set_register(0x00, 1, 1)
    """
    device_info: 'DeviceInfo'
    raw: pyxdaq_device.Device
    status: dict
    info: dict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.raw is not None:
            self.raw.close()
            self.raw = None


@dataclass
class DeviceInfo:
    """
    Metadata and configuration for a single XDAQ device discovered via scan.

    Attributes:
        manager_path: Filesystem path to the device manager shared object.
        manager_info: Metadata about the device manager.
        options: Device-specific configuration parameters (mutable).
        serial_number: Device serial number reported during scan.
        api_version: Firmware/API version reported during scan.
        info: Pre-fetched device info dict.
        status: Pre-fetched device status dict.
    """
    manager_path: Path
    manager_info: dict
    options: dict
    serial_number: str = ""
    api_version: str = ""
    info: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)

    def with_mode(self, mode: str) -> Self:
        """
        Return a copy of this DeviceInfo with the 'mode' option set.
        """
        info = deepcopy(self)
        info.options['mode'] = mode.lower()
        return info

    def create(self) -> Device:
        """
        Instantiate a Device using the current DeviceInfo.

        Returns:
            A Device object wrapping the active low-level device instance.
        """
        manager = pyxdaq_device.get_device_manager(self.manager_path)
        raw = manager.create_device(json.dumps(self.options))
        return Device(self, raw, json.loads(raw.get_status()), json.loads(raw.get_info()))


ScanResult = Union[
    pyxdaq_device.DeviceFound,
    pyxdaq_device.DeviceOccupied,
    pyxdaq_device.DeviceQueryFailed,
]


def scan_devices(manager_paths: List[Path]) -> List[ScanResult]:
    """
    Scan device manager plugins and return one result per enumerated device.

    Each result is one of:
        - ``DeviceFound``: reachable and not in use.
        - ``DeviceOccupied``: held open by another process.
        - ``DeviceQueryFailed``: visible but info/status query failed.

    Args:
        manager_paths: Plugin paths to scan

    Returns:
        List of scan results in discovery order.
    """
    return pyxdaq_device.scan_devices(manager_paths)


def list_devices(manager_dir: Optional[Path] = None) -> List[DeviceInfo]:
    """
    Enumerate all available (non-occupied) XDAQ devices.

    Uses the scanner internally so no device is opened just to read metadata.
    Occupied and failed devices are silently filtered out; use ``scan_devices``
    for the full picture.

    Args:
        manager_dir: Directory to scan for device manager plugins. Defaults to DeviceManagerDir.

    Returns:
        A sorted list of DeviceInfo instances describing each available device.
    """
    if manager_dir is None:
        manager_dir = DeviceManagerDir

    # Cache manager info per path to avoid reloading the same DLL multiple times.
    _manager_info_cache: dict = {}

    def _manager_info(path: Path) -> dict:
        if path not in _manager_info_cache:
            mgr = pyxdaq_device.get_device_manager(path)
            _manager_info_cache[path] = json.loads(mgr.info())
        return _manager_info_cache[path]

    devices = []
    for result in pyxdaq_device.scan_devices_dir(manager_dir):
        if not isinstance(result, pyxdaq_device.DeviceFound):
            continue
        devices.append(
            DeviceInfo(
                manager_path=result.device_manager_path,
                manager_info=_manager_info(result.device_manager_path),
                options=json.loads(result.device_config_json),
                serial_number=result.serial_number,
                api_version=result.api_version,
                info=json.loads(result.info_json),
                status=json.loads(result.status_json),
            )
        )

    return sorted(devices, key=lambda x: str(x.options))
