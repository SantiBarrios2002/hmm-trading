"""Core package for the HMM trading project."""

from hft_hmm._version import __version__
from hft_hmm.project import PROJECT_NAME, ProjectInfo, get_project_info

__all__ = ["PROJECT_NAME", "ProjectInfo", "__version__", "get_project_info"]
