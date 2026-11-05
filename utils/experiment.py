"""One source receipt for node agreement, checkpoints and experiment logs."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def source_hashes():
    paths=[ROOT/'launch_job.py',ROOT/'train_production.py',ROOT/'scripts/launch.sh']
    for folder in ('ops','distributed','models','data','training','utils','configs','evaluations'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.json','.yaml'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
