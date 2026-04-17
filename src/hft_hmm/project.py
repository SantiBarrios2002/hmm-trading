"""Project-level metadata helpers for scaffolding and tooling."""

from dataclasses import dataclass

PROJECT_NAME = "hft-hmm"


@dataclass(frozen=True)
class ProjectInfo:
    """Static metadata exposed for tooling and smoke tests."""

    name: str
    package: str
    python_requires: str


def get_project_info() -> ProjectInfo:
    """Return stable metadata for repository-level smoke checks."""

    return ProjectInfo(
        name=PROJECT_NAME,
        package="hft_hmm",
        python_requires=">=3.11,<3.12",
    )
