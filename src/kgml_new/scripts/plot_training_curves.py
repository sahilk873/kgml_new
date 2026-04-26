from __future__ import annotations

import argparse
from pathlib import Path

from kgml_new.training.plot_curves import plot_history_curves, plot_run_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train/val loss curves from a run JSON or inline history."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to run_gpu_method output JSON (or use --history-json for raw history only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Treat --input as a JSON object with only epoch/train_loss/val_loss keys.",
    )
    parser.add_argument(
        "--no-encoder",
        action="store_true",
        help="For full run JSON: skip base_embedding_history subplot.",
    )
    parser.add_argument(
        "--no-decoder",
        action="store_true",
        help="For full run JSON: skip decoder_history subplot.",
    )
    args = parser.parse_args()

    if args.history_only:
        import json

        hist = json.loads(Path(args.input).read_text())
        plot_history_curves(hist, args.output, title=args.output.stem)
    else:
        plot_run_json(
            args.input,
            args.output,
            include_encoder=not args.no_encoder,
            include_decoder=not args.no_decoder,
        )


if __name__ == "__main__":
    main()
