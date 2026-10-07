"""Load either the course grayscale HDF5 subset or the official SVHN MAT files."""

from pathlib import Path

import numpy as np


def _validate(images: np.ndarray, labels: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(images)
    labels = np.asarray(labels).reshape(-1)
    if images.ndim != 3 or images.shape[1:] != (32, 32):
        raise ValueError(f"{name}: expected grayscale images shaped (N, 32, 32)")
    if len(images) == 0 or len(images) != len(labels):
        raise ValueError(f"{name}: image and label counts must match and be nonempty")
    if not np.issubdtype(images.dtype, np.number) or not np.isfinite(images).all():
        raise ValueError(f"{name}: image pixels must be finite numbers")
    if images.min() < 0 or images.max() > 255:
        raise ValueError(f"{name}: expected pixel values in [0, 255]")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, np.arange(10)).all():
        raise ValueError(f"{name}: labels must be integers from 0 to 9")
    return images, labels.astype(np.int64, copy=False)


def load_course_h5(path: str | Path) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Use only X_train/y_train and X_test/y_test; the notebook's X_val is unused."""
    import h5py

    with h5py.File(path, "r") as source:
        required = {"X_train", "y_train", "X_test", "y_test"}
        missing = required - set(source.keys())
        if missing:
            raise ValueError(f"Missing HDF5 arrays: {sorted(missing)}")
        train = _validate(source["X_train"][:], source["y_train"][:], "train")
        test = _validate(source["X_test"][:], source["y_test"][:], "test")
    return train, test


def load_official_mat(train_path: str | Path, test_path: str | Path):
    """Read Stanford's cropped-digit MAT format, converting RGB to grayscale.

    This path uses different data and preprocessing from the course HDF5 file;
    its newly trained metrics must not be compared as a reproduction of that run.
    """
    from scipy.io import loadmat

    def read(path: str | Path, name: str):
        source = loadmat(path)
        if "X" not in source or "y" not in source:
            raise ValueError(f"{name}: MAT file needs X and y")
        rgb = np.asarray(source["X"])
        if rgb.ndim != 4 or rgb.shape[:3] != (32, 32, 3):
            raise ValueError(f"{name}: expected X shaped (32, 32, 3, N)")
        # Weighted luminance on the original 8-bit channels, keeping float pixels.
        gray = np.tensordot(rgb.astype(np.float32), [0.299, 0.587, 0.114], axes=([2], [0]))
        gray = np.moveaxis(gray, -1, 0)
        labels = np.asarray(source["y"]).reshape(-1)
        labels = np.where(labels == 10, 0, labels)  # SVHN encodes digit zero as 10.
        return _validate(gray, labels, name)

    return read(train_path, "train"), read(test_path, "test")


def prepare_images(images: np.ndarray, model_name: str) -> np.ndarray:
    """Normalize to [0, 1], adding a channel dimension for CNNs."""
    result = np.asarray(images, dtype=np.float32) / 255.0
    if model_name.startswith("cnn_"):
        result = result[..., np.newaxis]
    return result

