"""PaperBanana package init — ensures the package root is on sys.path
so that bare imports like 'from utils.xxx import ...' resolve correctly
when deployed as a renamed package on Agent Engine."""

import sys
from pathlib import Path

_pkg_root = str(Path(__file__).resolve().parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
