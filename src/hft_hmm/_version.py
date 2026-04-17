"""Package version metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hft-hmm")
except PackageNotFoundError:
    __version__ = "0.1.0"
