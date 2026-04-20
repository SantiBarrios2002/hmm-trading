"""Experiment configuration dataclasses, deterministic YAML I/O, and run-id hashing.

The paper (§4.4) reports a single 1-minute ES simulation; this module makes any
such experiment re-executable from a saved YAML. ``ExperimentConfig`` bundles a
typed ``DataSourceConfig`` with the :class:`WalkForwardConfig` already defined in
the experiments layer, plus frequency / cost / notes metadata. A canonical YAML
representation is used as the hash input for :func:`run_id`, so two configs hash
to the same 12-char id iff they produce the same canonical bytes.

References: §4.4 reproducible simulation artifacts (evaluation layer)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from hft_hmm.core import EVALUATION_LAYER, PaperReference, reference
from hft_hmm.experiments.walk_forward import WalkForwardConfig

__category__: Final[str] = EVALUATION_LAYER
EXPERIMENT_CONFIG_REFERENCE: Final[PaperReference] = reference(
    "§4.4", "reproducible simulation artifacts"
)

DataSourceKind = Literal["csv", "databento_parquet", "yfinance"]
_DATA_SOURCE_KINDS: Final[tuple[str, ...]] = ("csv", "databento_parquet", "yfinance")
_REPRODUCIBLE_KINDS: Final[frozenset[str]] = frozenset({"csv", "databento_parquet"})

Frequency = Literal["1min", "5min", "1D"]
_FREQUENCIES: Final[tuple[str, ...]] = ("1min", "5min", "1D")


@dataclass(frozen=True)
class DataSourceConfig:
    """Typed description of where a run's raw market data comes from.

    ``csv`` and ``databento_parquet`` point at local, immutable files and are
    treated as reproducible. ``yfinance`` is a vendor feed that may revise
    history across calls and is flagged non-reproducible.
    """

    kind: DataSourceKind
    path: str | None = None
    symbol: str | None = None
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _DATA_SOURCE_KINDS:
            raise ValueError(f"kind must be one of {_DATA_SOURCE_KINDS}, got {self.kind!r}.")
        if self.kind == "csv":
            if self.path is None:
                raise ValueError("DataSourceConfig(kind='csv') requires path.")
        elif self.kind == "databento_parquet":
            if self.path is None:
                raise ValueError("DataSourceConfig(kind='databento_parquet') requires path.")
            if self.symbol is None:
                raise ValueError("DataSourceConfig(kind='databento_parquet') requires symbol.")
        else:  # yfinance
            if self.path is not None:
                raise ValueError(
                    "DataSourceConfig(kind='yfinance') must not set path; "
                    "yfinance pulls from a remote API."
                )
            if self.symbol is None:
                raise ValueError("DataSourceConfig(kind='yfinance') requires symbol.")
            if self.start is None or self.end is None:
                raise ValueError(
                    "DataSourceConfig(kind='yfinance') requires start and end ISO dates."
                )

    @property
    def is_reproducible(self) -> bool:
        """``True`` when the source is a local immutable file, ``False`` for yfinance."""
        return self.kind in _REPRODUCIBLE_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "symbol": self.symbol,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataSourceConfig:
        return cls(
            kind=data["kind"],
            path=data.get("path"),
            symbol=data.get("symbol"),
            start=data.get("start"),
            end=data.get("end"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Full recipe for one walk-forward experiment run.

    ``notes`` is part of the hash so two otherwise-identical configs with
    different notes resolve to different run ids — useful when iterating on
    human-readable experiment descriptions without touching numerical fields.
    """

    data: DataSourceConfig
    frequency: Frequency
    walk_forward: WalkForwardConfig
    cost_bps_per_turnover: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.data, DataSourceConfig):
            raise TypeError(f"data must be a DataSourceConfig, got {type(self.data).__name__}.")
        if not isinstance(self.walk_forward, WalkForwardConfig):
            raise TypeError(
                f"walk_forward must be a WalkForwardConfig, got {type(self.walk_forward).__name__}."
            )
        if self.frequency not in _FREQUENCIES:
            raise ValueError(f"frequency must be one of {_FREQUENCIES}, got {self.frequency!r}.")
        if not isinstance(self.cost_bps_per_turnover, (int, float)) or isinstance(
            self.cost_bps_per_turnover, bool
        ):
            raise TypeError(
                "cost_bps_per_turnover must be a real number, got "
                f"{type(self.cost_bps_per_turnover).__name__}."
            )
        if self.cost_bps_per_turnover < 0.0:
            raise ValueError(
                "cost_bps_per_turnover must be non-negative, "
                f"got {self.cost_bps_per_turnover!r}."
            )
        if not isinstance(self.notes, str):
            raise TypeError(f"notes must be a str, got {type(self.notes).__name__}.")

    def to_dict(self) -> dict[str, Any]:
        # retrain_every_days is always a concrete int after WalkForwardConfig.__post_init__
        # normalizes the None default; the Optional in the type declaration is just so the
        # default can be left unset at construction time.
        retrain_every_days = self.walk_forward.retrain_every_days
        assert retrain_every_days is not None
        return {
            "data": self.data.to_dict(),
            "frequency": self.frequency,
            "cost_bps_per_turnover": float(self.cost_bps_per_turnover),
            "walk_forward": {
                "h_days": int(self.walk_forward.h_days),
                "t_days": int(self.walk_forward.t_days),
                "retrain_every_days": int(retrain_every_days),
                "k_values": list(self.walk_forward.k_values),
                "random_state": int(self.walk_forward.random_state),
                "n_iter": int(self.walk_forward.n_iter),
                "tol": float(self.walk_forward.tol),
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        wf_raw = data["walk_forward"]
        walk_forward = WalkForwardConfig(
            h_days=int(wf_raw["h_days"]),
            t_days=int(wf_raw["t_days"]),
            retrain_every_days=_optional_int(wf_raw.get("retrain_every_days")),
            k_values=tuple(int(k) for k in wf_raw["k_values"]),
            random_state=int(wf_raw["random_state"]),
            n_iter=int(wf_raw["n_iter"]),
            tol=float(wf_raw["tol"]),
        )
        return cls(
            data=DataSourceConfig.from_dict(data["data"]),
            frequency=data["frequency"],
            walk_forward=walk_forward,
            cost_bps_per_turnover=float(data.get("cost_bps_per_turnover", 0.0)),
            notes=str(data.get("notes", "")),
        )

    def to_yaml_bytes(self) -> bytes:
        """Canonical UTF-8 YAML bytes — the exact input to :func:`run_id`."""
        return yaml.safe_dump(
            self.to_dict(),
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=False,
            width=1_000_000,
        ).encode("utf-8")

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_yaml_bytes())

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"YAML at {path} must decode to a mapping, got {type(raw).__name__}.")
        return cls.from_dict(raw)


def run_id(config: ExperimentConfig) -> str:
    """Return the 12-char hex run id derived from the config's canonical YAML bytes.

    Two configs hash to the same id iff they produce the same canonical YAML
    output, so any field change (including ``notes``) changes the id. The
    12-char prefix is collision-resistant enough for human-scale experiment
    directories while staying short in filesystem paths.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError(f"config must be an ExperimentConfig, got {type(config).__name__}.")
    return hashlib.sha256(config.to_yaml_bytes()).hexdigest()[:12]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
