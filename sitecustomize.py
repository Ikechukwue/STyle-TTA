from pathlib import Path
import sys

root = Path(__file__).resolve().parent
code_root = root / "code"

for candidate in (str(code_root), str(root)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
