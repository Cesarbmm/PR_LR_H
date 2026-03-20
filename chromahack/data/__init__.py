from .dataset_io import DatasetSample, DatasetStats, load_dataset_payload, save_dataset_payload

__all__ = [
    "DatasetSample",
    "DatasetStats",
    "SyntheticDatasetGenerator",
    "load_dataset_payload",
    "save_dataset_payload",
]


def __getattr__(name: str):
    if name == "SyntheticDatasetGenerator":
        from .generate_dataset import SyntheticDatasetGenerator

        return SyntheticDatasetGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
