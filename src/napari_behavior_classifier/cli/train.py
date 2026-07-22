"""Command-line training: pool a session's annotations into one model, print a report.

    napari-behavior-train session.json model.joblib

Mirrors the widget's "Train + Predict" button (see `_widget.py`'s
`_TrainingSummaryDialog`) but for headless/scripted use - same pooling-across-
individuals semantics (`api.train_from_session`), same out-of-bag report, printed
as plain text instead of a Qt dialog.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import api


def format_report(result: api.TrainResult) -> str:
    lines = [
        f"Individuals (pooled): {', '.join(result.individuals)}",
        f"Files: {', '.join(result.files)}",
        f"Labeled frames: {sum(result.class_counts.values())}  |  Features: {result.n_features}",
    ]

    report = result.oob_report
    if report is None:
        lines.append("Out-of-bag accuracy: n/a (not enough labeled frames per class)")
        return "\n".join(lines)

    lines.append(f"Out-of-bag accuracy: {report.accuracy:.1%}")
    lines.append("")

    header = ["Class", "Annotated", "Precision", "Recall", "F1"] + [f"-> {c}" for c in report.classes]
    rows = [header]
    for i, cls in enumerate(report.classes):
        rows.append(
            [
                cls,
                str(result.class_counts.get(cls, 0)),
                f"{report.precision[i]:.2f}",
                f"{report.recall[i]:.2f}",
                f"{report.f1[i]:.2f}",
            ]
            + [str(n) for n in report.confusion[i]]
        )

    widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one pooled behavior classifier from a saved session.")
    parser.add_argument("session", type=Path, help="Session .json file (open files + annotations) saved from the GUI")
    parser.add_argument("model_out", type=Path, help="Where to write the trained model (.joblib)")
    args = parser.parse_args(argv)

    try:
        result = api.train_from_session(args.session)
    except ValueError as e:
        print(f"Training failed: {e}", file=sys.stderr)
        return 1

    api.save_pipeline(result.pipeline, args.model_out)

    print(format_report(result))
    print(f"\nSaved model to {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
