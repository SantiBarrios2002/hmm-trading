"""Smoke tests for the repository scaffold."""

from hft_hmm import PROJECT_NAME, __version__, get_project_info


def test_package_metadata_matches_project_configuration() -> None:
    info = get_project_info()

    assert PROJECT_NAME == "hft-hmm"
    assert info.name == PROJECT_NAME
    assert info.package == "hft_hmm"
    assert info.python_requires == ">=3.11,<3.12"


def test_package_exposes_version_string() -> None:
    assert __version__ == "0.1.0"
