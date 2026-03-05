from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

from .config import PreprocessConfig
from .extractors import (
    OmniEraserBackgroundExtractor,
    SamBody4DMaskletExtractor,
    SubjectPortraitExtractor,
)
from .io.video import read_frames


def _sanitize_sample_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return cleaned or "sample"


def _contains_image_sequence(path: Path) -> bool:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return any(p.is_file() and p.suffix.lower() in image_exts for p in path.iterdir())


def _discover_inputs(config: PreprocessConfig) -> list[Path]:
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input_path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    if _contains_image_sequence(input_path):
        return [input_path]

    matcher = input_path.rglob if config.recursive else input_path.glob
    videos = [p for p in matcher("*") if p.is_file() and p.suffix.lower() in config.video_extensions]
    if videos:
        return sorted(videos)

    # Fallback: treat child directories with frame sequences as independent samples.
    frame_dirs = [p for p in matcher("*") if p.is_dir() and _contains_image_sequence(p)]
    return sorted(frame_dirs)


def _relative(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


class DataPreprocessingPipeline:
    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self.output_root = Path(config.output_root)
        self.samples_root = self.output_root / "samples"
        if not config.sam_body4d.command:
            raise ValueError("sam_body4d.command is required.")
        if not config.omni_eraser.command:
            raise ValueError("omni_eraser.command is required.")

        self.masklet_extractor = SamBody4DMaskletExtractor(config.sam_body4d.command)
        self.background_extractor = OmniEraserBackgroundExtractor(config.omni_eraser.command)

        self.portrait_extractor = SubjectPortraitExtractor(margin=config.portrait_margin)

    def _materialize_target_input(self, src: Path, dst_dir: Path) -> tuple[str, Path]:
        if src.is_file():
            dst = dst_dir / f"target_video{src.suffix.lower()}"
            if self.config.copy_target_video:
                shutil.copy2(src, dst)
            else:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            return "target_video", dst

        dst = dst_dir / "target_frames"
        if self.config.copy_target_video:
            shutil.copytree(src, dst)
        else:
            if dst.exists():
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            dst.symlink_to(src.resolve(), target_is_directory=True)
        return "target_frames_dir", dst

    def run(self) -> list[dict]:
        inputs = _discover_inputs(self.config)
        if not inputs:
            raise RuntimeError(
                f"No valid inputs found under {self.config.input_path} "
                f"(video extensions: {self.config.video_extensions}, or image-sequence directories)."
            )

        self.samples_root.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        seen_ids: set[str] = set()

        if self.config.dry_run:
            print(f"[dry-run] found {len(inputs)} inputs")
            for path in inputs:
                print(f"[dry-run] {path}")
            return records

        for index, input_path in enumerate(inputs):
            sample_id = _sanitize_sample_id(input_path.stem if input_path.is_file() else input_path.name)
            if sample_id in seen_ids:
                sample_id = f"{sample_id}_{index:04d}"
            seen_ids.add(sample_id)

            sample_dir = self.samples_root / sample_id
            if sample_dir.exists():
                if not self.config.overwrite:
                    print(f"[skip] sample exists: {sample_dir}")
                    continue
                shutil.rmtree(sample_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)

            target_key, target_path = self._materialize_target_input(input_path, sample_dir)
            frames = read_frames(input_path, max_frames=self.config.max_frames)
            if len(frames) == 0:
                raise RuntimeError(f"No decoded frames for input: {input_path}")

            masklet_dir = sample_dir / "masklets"
            masks = self.masklet_extractor.extract(input_path, frames, masklet_dir, self.config)
            if masks.ndim != 3:
                raise RuntimeError(f"Masklets must be T,H,W for {input_path}, got {masks.shape}")

            t = min(len(frames), masks.shape[0])
            frames = frames[:t]
            masks = masks[:t]
            if t == 0:
                raise RuntimeError(f"After alignment, no frames remain: {input_path}")

            bg_path = sample_dir / "background.png"
            self.background_extractor.extract(input_path, frames, masks, masklet_dir, bg_path)

            portrait_path = sample_dir / "subject_portrait.png"
            self.portrait_extractor.extract(frames, masks, portrait_path)

            metadata = {
                "sample_id": sample_id,
                "num_frames": int(t),
                "height": int(np.asarray(frames[0]).shape[0]),
                "width": int(np.asarray(frames[0]).shape[1]),
                target_key: _relative(target_path, self.output_root),
                "ref_portrait": _relative(portrait_path, self.output_root),
                "background_image": _relative(bg_path, self.output_root),
                "masklets_dir": _relative(masklet_dir, self.output_root),
            }

            metadata_path = sample_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(metadata)
            print(f"[ok] {sample_id} frames={t}")

        dataset_path = self.output_root / "dataset.jsonl"
        with dataset_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[done] wrote {len(records)} samples -> {dataset_path}")
        return records
