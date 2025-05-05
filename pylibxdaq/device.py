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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Self

from . import pyxdaq_device
from .managers import DeviceManagerPaths

__all__ = ["DeviceInfo", "Device", "list_devices"]


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
        self.raw = None  # ensure resource release


@dataclass
class DeviceInfo:
    """
    Metadata and configuration for a single XDAQ device.

    Attributes:
        manager_path: Filesystem path to the device manager shared object.
        manager_info: Metadata about the device manager.
        options: Device-specific configuration parameters.
    """
    manager_path: Path
    manager_info: dict
    options: dict

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
        manager = pyxdaq_device.get_device_manager(str(self.manager_path))
        raw = manager.create_device(json.dumps(self.options))
        return Device(self, raw, json.loads(raw.get_status()), json.loads(raw.get_info()))


def list_devices(manager_paths: Optional[List[Path]] = None) -> List[DeviceInfo]:
    """
    Enumerate all available XDAQ devices from the given manager paths.

    Args:
        manager_paths: Optional list of shared object paths. Defaults to DeviceManagerPaths.

    Returns:
        A sorted list of DeviceInfo instances describing each available device.
    """
    if manager_paths is None:
        manager_paths = DeviceManagerPaths

    devices = []
    for path in manager_paths:
        manager = pyxdaq_device.get_device_manager(str(path))
        manager_info = json.loads(manager.info())
        device_list = json.loads(manager.list_devices())
        for options in device_list:
            devices.append(DeviceInfo(path, manager_info, options))

    return sorted(devices, key=lambda x: str(x.options))
