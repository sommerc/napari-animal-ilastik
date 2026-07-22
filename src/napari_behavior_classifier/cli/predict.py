"""Command-line prediction: run a trained model over one or more .h5 files.

    napari-behavior-predict model.joblib file1.h5 file2.h5 ...

Writes one CSV per input .h5 (next to it by default), with the same
[File, Animal_ID, Frame, Class] columns as the GUI's "Export predictions" button.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import api


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict behavior classes for one or more SLEAP analysis .h5 files.")
    parser.add_argument("model", type=Path, help="Trained model file (.joblib)")
    parser.add_argument("h5_paths", type=Path, nargs="+", help="One or more SLEAP analysis .h5 files")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write CSVs into (default: alongside each input .h5)",
    )
    args = parser.parse_args(argv)

    pipeline = api.load_pipeline(args.model)

    for h5_path in args.h5_paths:
        predictions = api.predict_file_all_individuals(pipeline, h5_path)
        rows = [
            (h5_path.name, individual, frame, class_name)
            for individual, preds in predictions.items()
            for frame, class_name in enumerate(preds)
        ]
        df = pd.DataFrame(rows, columns=["File", "Animal_ID", "Frame", "Class"])

        out_dir = args.out_dir if args.out_dir is not None else h5_path.parent
        out_path = out_dir / (h5_path.stem + "_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"{h5_path.name}: {len(df)} predicted frames -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
