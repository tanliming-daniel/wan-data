from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_preprocess_config
from .pipeline import DataPreprocessingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build WAN controllable-video preprocessing dataset: target video + portrait + background + masklets."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to yaml/json config.")
    parser.add_argument("--input", type=str, default=None, help="Override input_path.")
    parser.add_argument("--output", type=str, default=None, help="Override output_root.")
    parser.add_argument(
        "--masklet-source",
        type=str,
        default=None,
        choices=["sam_body4d"],
        help="Override masklet extractor source.",
    )
    parser.add_argument(
        "--background-source",
        type=str,
        default=None,
        choices=["omni_eraser"],
        help="Override background extractor source.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Override max_frames.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample folders.")
    parser.add_argument("--dry-run", action="store_true", help="Only list discovered videos.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_preprocess_config(args.config)

    if args.input is not None:
        cfg.input_path = str(Path(args.input).expanduser().resolve())
    if args.output is not None:
        cfg.output_root = str(Path(args.output).expanduser().resolve())
    if args.masklet_source is not None:
        cfg.masklet_source = args.masklet_source
    if args.background_source is not None:
        cfg.background_source = args.background_source
    if args.max_frames is not None and args.max_frames > 0:
        cfg.max_frames = args.max_frames
    if args.overwrite:
        cfg.overwrite = True
    if args.dry_run:
        cfg.dry_run = True

    cfg.normalize()
    pipeline = DataPreprocessingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
