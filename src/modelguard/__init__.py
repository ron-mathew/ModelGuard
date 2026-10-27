"""ModelGuard package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("modelguard")
except PackageNotFoundError:
    __version__ = "0.1.0"
