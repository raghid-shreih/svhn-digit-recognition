# SVHN Digit Recognition

A computer vision case study comparing dense neural networks with convolutional neural networks for recognizing single digits in 32 × 32 street-view crops. It translates a completed MIT Applied AI course experiment into a small, runnable Python project with explicit data loading, model definitions, offline checks, and carefully labeled results.

## Results at a glance

The completed course notebook recorded a **deeper CNN at 16,162/18,000 correct (89.79%)**, compared with **13,399/18,000 (74.44%)** for a deeper dense neural network (ANN) on its grayscale course subset. These values were independently computed from the notebook's printed test confusion matrices; **the models have not been retrained in this repository**.

![Comparison of dense and convolutional models on the course subset: 74.44 percent versus 89.79 percent](assets/model-comparison.svg)

See the [results note](docs/results.md) for the CNN's ten-class confusion matrix and interpretation limits. The public code makes it possible to run a new experiment, which may produce different metrics.

## What this demonstrates

- Loading and validating image arrays and labels from HDF5 or official SVHN MAT files.
- Preparing grayscale inputs for dense and convolutional classifiers.
- Building two dense and two CNN architectures in TensorFlow/Keras, including dropout and batch normalization in the deeper variants.
- Keeping an explicit validation split within training data, then reporting ten-class test metrics.
- Checking a written result against its confusion matrix and communicating where the evidence stops.

## Run locally

Use Python **3.10–3.12** (3.11 is a good default). TensorFlow training is an optional, relatively large install. A GPU is helpful for the full dataset, but the code also runs on a supported CPU setup.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
python -m unittest discover -s tests -v
```

### Course grayscale HDF5 subset

If you have an authorized copy of `SVHN_single_grey1.h5`, place it in the ignored `data/` folder and run:

```bash
svhn-recognition --h5 data/SVHN_single_grey1.h5 --model cnn_regularized --epochs 30 --output results/course-cnn.json
```

The runner uses `X_train` and `X_test`, holds out a **seeded stratified 20% of training** for validation, and evaluates the test set after training. The supplied notebook instead used Keras's `validation_split=0.2`; its separately loaded 60,000-image `X_val` array was not used. This runner is a **new experiment**, not an exact recreation of the historical 89.79% run.

### Official SVHN cropped digits

You can also obtain `train_32x32.mat` and `test_32x32.mat` from the [Stanford SVHN dataset page](http://ufldl.stanford.edu/housenumbers/) and place them in `data/`:

```bash
svhn-recognition --official-train data/train_32x32.mat --official-test data/test_32x32.mat --model cnn_regularized --epochs 30 --output results/official-cnn.json
```

This path converts RGB images to grayscale and maps the source label **10** to digit **0**. The official source files and preprocessing differ from the course subset, so compare new results only with experiments on the same data and split. The `extra` SVHN split is not used.

For a dense-network comparison, choose `--model ann_deep` and a separate output path. Available architectures are `ann_baseline`, `ann_deep`, `cnn_baseline`, and `cnn_regularized`. You can reduce `--epochs` for a smoke run. Seeds improve repeatability but do not guarantee identical results across hardware or library versions.

The CLI saves aggregate metrics, the confusion matrix, class precision/recall/F1, and training history to the ignored `results/` directory. Select hyperparameters with validation data and reserve test data for the final assessment.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/svhn_recognition/` | Loaders, model definitions, training and evaluation CLI |
| `tests/` | Small synthetic offline checks, no dataset or GPU needed |
| `docs/results.md` | Checked notebook observations and limits |
| `docs/historical_metrics.json` | Aggregate matrices transcribed from printed notebook output |
| `scripts/build_historical_charts.py` | Rebuild the aggregate SVGs shown here |
| `assets/` | Model comparison and ten-digit error matrix |

## Data and provenance

The course-provided grayscale HDF5 file, original assignment notebook, trained models, and extracted digit images are **not distributed here**. The original notebook contains course instructions, and redistribution rights for the processed file have not been established. Source dataset information is available from [Stanford](http://ufldl.stanford.edu/housenumbers/) and the [SVHN paper](https://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf). The code in this repository is newly organized for this standalone case study; no license is asserted for the source dataset or course materials.
