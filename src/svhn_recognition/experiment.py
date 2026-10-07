"""Train on a local dataset, with a train-only validation split."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .data import prepare_images
from .models import build_model


def summarize_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Expected nonempty, equal-length labels and predictions")
    if not np.isin(actual, np.arange(10)).all() or not np.isin(predicted, np.arange(10)).all():
        raise ValueError("Labels and predictions must be digits 0-9")
    matrix = confusion_matrix(actual, predicted, labels=list(range(10)))
    return {"tested": int(len(actual)), "correct": int(np.trace(matrix)),
            "accuracy": float(np.trace(matrix) / matrix.sum()),
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(actual, predicted,
                labels=list(range(10)), output_dict=True, zero_division=0)}


def run_experiment(train: tuple, test: tuple, *, model_name: str, output: str | Path,
                   epochs: int, batch_size: int = 128, seed: int = 42) -> dict:
    """Fit a model and evaluate the held-out test set once after training.

    Model/epoch choices should be made using validation data, before inspecting
    the test result. This run does not reproduce the notebook's exact split.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    images, labels = train
    test_images, test_labels = test
    train_idx, validation_idx = train_test_split(
        np.arange(len(labels)), test_size=.2, random_state=seed, stratify=labels)
    x = prepare_images(images, model_name)
    x_test = prepare_images(test_images, model_name)
    model = build_model(model_name)
    history = model.fit(x[train_idx], labels[train_idx],
                        validation_data=(x[validation_idx], labels[validation_idx]),
                        epochs=epochs, batch_size=batch_size, verbose=2)
    predictions = np.argmax(model.predict(x_test, batch_size=batch_size, verbose=0), axis=1)
    report = {"model": model_name, "seed": seed, "epochs": epochs,
              "batch_size": batch_size, "train_examples": len(train_idx),
              "validation_examples": len(validation_idx),
              "history": {key: [float(v) for v in values] for key, values in history.history.items()},
              "test": summarize_predictions(test_labels, predictions)}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

