try:
    from ._version import version as __version__, version_tuple  # type: ignore
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from pathlib import Path

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent
