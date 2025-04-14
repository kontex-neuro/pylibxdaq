from pathlib import Path

manager_paths = list(
    {
        p.resolve()
        for p in Path(__file__).parent.glob('*')
        if (p.is_file() and 'device_manager' in str(p))
    }
)