from datetime import datetime
from pathlib import Path


def create_run_dir(base_dir="experiments/training"):
    """
    Creates and returns a new run directory:
        <base_dir>/<dd.mm.yy>/run1/
        <base_dir>/<dd.mm.yy>/run2/
    """
    date_str = datetime.now().strftime("%d.%m.%y")
    date_dir = Path(base_dir) / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    existing_nums = []
    for p in date_dir.iterdir():
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit():
            existing_nums.append(int(p.name[3:]))

    next_num = max(existing_nums, default=0) + 1
    run_dir = date_dir / f"run{next_num}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir