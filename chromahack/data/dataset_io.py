"""Dataset serialization helpers with legacy pickle compatibility."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import numpy as np


@dataclass
class DatasetSample:
    frame: np.ndarray
    label: int
    true_order_pct: float
    n_objects: int
    variant: str
    seed: int


@dataclass
class DatasetStats:
    n_ordered: int = 0
    n_disordered: int = 0
    n_partial: int = 0
    n_augmented: int = 0
    mean_true_ordered: float = 0.0
    mean_true_disordered: float = 0.0
    fragility_level: str = "medium"
    notes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


_SAMPLE_FIELDS = {field.name for field in fields(DatasetSample)}
_STATS_FIELDS = {field.name for field in fields(DatasetStats)}


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler that remaps legacy DatasetSample/DatasetStats classes."""

    def find_class(self, module: str, name: str):
        if name == "DatasetSample":
            return DatasetSample
        if name == "DatasetStats":
            return DatasetStats
        return super().find_class(module, name)


def _coerce_mapping_record(obj: Any, allowed_fields: set[str]) -> dict | None:
    if obj is None:
        return None
    if is_dataclass(obj):
        raw = asdict(obj)
    elif isinstance(obj, dict):
        raw = dict(obj)
    else:
        raw = {field: getattr(obj, field) for field in allowed_fields if hasattr(obj, field)}

    if not raw:
        return None
    return {field: raw.get(field) for field in allowed_fields}


def sample_to_record(sample: Any) -> dict | None:
    record = _coerce_mapping_record(sample, _SAMPLE_FIELDS)
    if record is None:
        return None
    record["label"] = int(record["label"])
    record["true_order_pct"] = float(record["true_order_pct"])
    record["n_objects"] = int(record["n_objects"])
    record["variant"] = str(record["variant"])
    record["seed"] = int(record["seed"])
    return record


def stats_to_record(stats: Any) -> dict:
    record = _coerce_mapping_record(stats, _STATS_FIELDS) or {}
    notes = record.get("notes") or []
    record.setdefault("n_ordered", 0)
    record.setdefault("n_disordered", 0)
    record.setdefault("n_partial", 0)
    record.setdefault("n_augmented", 0)
    record.setdefault("mean_true_ordered", 0.0)
    record.setdefault("mean_true_disordered", 0.0)
    record.setdefault("fragility_level", "unknown")
    record["notes"] = list(notes)
    return record


def load_compat_pickle(path: str):
    with open(path, "rb") as handle:
        return _CompatUnpickler(handle).load()


def normalize_dataset_payload(payload: dict) -> dict:
    frames = list(payload.get("frames", []))
    labels = [int(label) for label in payload.get("labels", [])]
    sample_records = [sample_to_record(sample) for sample in payload.get("samples", [])]
    stats_record = stats_to_record(payload.get("stats", {}))

    if not frames and sample_records:
        frames = [sample["frame"] for sample in sample_records if sample is not None]
    if not labels and sample_records:
        labels = [max(0, int(sample["label"])) for sample in sample_records if sample is not None]
    if sample_records and len(sample_records) != len(frames):
        sample_records = []
    return {
        "frames": frames,
        "labels": labels,
        "samples": sample_records,
        "stats": stats_record,
    }


def load_dataset_payload(path: str) -> dict:
    payload = load_compat_pickle(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported dataset payload type: {type(payload)!r}")
    return normalize_dataset_payload(payload)


def save_dataset_payload(
    path: str,
    *,
    frames: list,
    labels: list[int],
    samples: list[Any] | None = None,
    stats: Any | None = None,
) -> dict:
    payload = {
        "frames": list(frames),
        "labels": [int(label) for label in labels],
        "samples": [sample_to_record(sample) for sample in (samples or [])],
        "stats": stats_to_record(stats or {}),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return payload


def save_stats_json(path: str, stats: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stats_to_record(stats), handle, indent=2)
