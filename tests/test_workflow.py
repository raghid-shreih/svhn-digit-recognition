import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

try:
    import h5py
except ImportError:
    h5py = None

from svhn_recognition.data import load_course_h5, load_official_mat, prepare_images
from svhn_recognition.experiment import summarize_predictions


class OfflineWorkflowTests(unittest.TestCase):
    @unittest.skipUnless(h5py is not None, "h5py is not installed in this environment")
    def test_h5_loader_ignores_extra_array_and_normalizes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.h5"
            with h5py.File(path, "w") as file:
                file["X_train"] = np.full((2, 32, 32), 255, dtype=np.uint8)
                file["y_train"] = [0, 1]
                file["X_test"] = np.zeros((1, 32, 32), dtype=np.uint8)
                file["y_test"] = [9]
                file["X_val"] = np.full((3, 32, 32), 17, dtype=np.uint8)
            train, test = load_course_h5(path)
            self.assertEqual(train[0].shape, (2, 32, 32))
            self.assertEqual(test[0].shape, (1, 32, 32))
            self.assertEqual(prepare_images(train[0], "cnn_regularized").shape, (2, 32, 32, 1))
            self.assertEqual(float(prepare_images(train[0], "ann_deep").max()), 1.0)

    def test_official_mat_remaps_zero_and_converts_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            train = Path(folder) / "train.mat"
            test = Path(folder) / "test.mat"
            rgb = np.zeros((32, 32, 3, 2), dtype=np.uint8)
            rgb[:, :, 0, 0] = 255
            rgb[:, :, 1, 1] = 255
            savemat(train, {"X": rgb, "y": np.array([[10], [1]])})
            savemat(test, {"X": rgb[:, :, :, :1], "y": np.array([[10]])})
            training, testing = load_official_mat(train, test)
            self.assertEqual(training[1].tolist(), [0, 1])
            self.assertEqual(testing[1].tolist(), [0])
            self.assertAlmostEqual(float(training[0][0, 0, 0]), 255*.299, places=3)

    def test_confusion_metrics_use_fixed_digit_order(self):
        report = summarize_predictions([0, 1, 1], [0, 0, 1])
        self.assertEqual((report["tested"], report["correct"]), (3, 2))
        self.assertEqual(report["confusion_matrix"][1][:2], [1, 1])
        self.assertEqual(len(report["confusion_matrix"]), 10)
        with self.assertRaises(ValueError):
            summarize_predictions([0], [10])


if __name__ == "__main__":
    unittest.main()
