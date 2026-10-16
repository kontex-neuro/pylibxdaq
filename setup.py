from setuptools import setup
import platform
import sys
import sysconfig

plat_name = sysconfig.get_platform()
if plat_name.endswith("universal2"):
    # python.org macOS builds report universal2, but the wheel holds only this machine's arch.
    # arm64 tags start at macOS 11; delocate raises the version to what the binaries need.
    plat_name = f"macosx-11.0-{platform.machine()}"

setup(
    options={
        "bdist_wheel":
            {
                "plat_name": plat_name,
                "python_tag": f"py{sys.version_info.major}{sys.version_info.minor}",
            },
    },
)
