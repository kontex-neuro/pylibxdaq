from setuptools import setup
import sys
import sysconfig

setup(
    options={
        "bdist_wheel":
            {
                "plat_name": sysconfig.get_platform(),
                "python_tag": f"py{sys.version_info.major}{sys.version_info.minor}",
            },
    },
)
