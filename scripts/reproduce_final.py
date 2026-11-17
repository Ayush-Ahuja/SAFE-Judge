"""Compatibility entry point. Use scripts/run_safe_judge.py for new commands."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_safe_judge import main

if __name__ == '__main__':
    if not any(arg=='--method' or arg.startswith('--method=') for arg in sys.argv[1:]):
        sys.argv.extend(['--method', 'selected_local'])
    main()
