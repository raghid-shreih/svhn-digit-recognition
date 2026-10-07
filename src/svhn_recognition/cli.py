"""Command-line interface for local SVHN experiments."""

import argparse

from .data import load_course_h5, load_official_mat
from .experiment import run_experiment


MODELS = ("ann_baseline", "ann_deep", "cnn_baseline", "cnn_regularized")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate a SVHN digit classifier")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--h5", help="Private grayscale course HDF5 file (not distributed)")
    source.add_argument("--official-train", help="Official SVHN train_32x32.mat")
    parser.add_argument("--official-test", help="Official SVHN test_32x32.mat; required with --official-train")
    parser.add_argument("--model", choices=MODELS, default="cnn_regularized")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/run.json")
    args = parser.parse_args(argv)
    if args.h5 and args.official_test or args.official_train and not args.official_test:
        parser.error("--official-test is required with --official-train and cannot be combined with --h5")
    train, test = (load_course_h5(args.h5) if args.h5 else
                   load_official_mat(args.official_train, args.official_test))
    report = run_experiment(train, test, model_name=args.model, output=args.output,
                            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    print(f"{report['model']}: {report['test']['correct']}/{report['test']['tested']} "
          f"({report['test']['accuracy']:.2%}) on held-out test data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

