from pathlib import Path

DeviceManagerPaths = list(
    {
        p.resolve()
        for p in Path(__file__).parent.glob('*')
        if (p.is_file() and 'device_manager' in str(p))
    }
)