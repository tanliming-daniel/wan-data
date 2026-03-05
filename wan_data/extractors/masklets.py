from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image

from ..config import PreprocessConfig


@dataclass
class SamBody4DExtraction:
    masks: np.ndarray
    mhr_path: Path
    sam3_first_mask_path: Path | None = None
    subject_mask_dirs: dict[int, Path] | None = None
    subject_manifest_path: Path | None = None


def _save_masks(masks: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        out_path = output_dir / f"{idx:06d}.png"
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def _load_mask_images(mask_dir: Path) -> np.ndarray:
    paths = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not paths:
        raise RuntimeError(f"No mask images found in {mask_dir}")
    masks = []
    for path in paths:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
        masks.append(mask)
    return np.stack(masks, axis=0)


def _discover_mhr_file(search_root: Path) -> Path | None:
    exts = {".npz", ".npy", ".pt", ".pth", ".json", ".jsonl", ".pkl"}
    candidates: list[Path] = []
    for path in search_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        name = path.name.lower()
        if "mhr" in name or "skeleton" in name or "rig" in name:
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            0 if "mhr" in p.name.lower() else 1,
            0 if p.parent == search_root else 1,
            len(str(p)),
        )
    )
    return candidates[0]


def _discover_subject_mask_dirs(subject_root: Path) -> dict[int, Path]:
    if not subject_root.exists():
        return {}
    mapping: dict[int, Path] = {}
    for path in sorted(subject_root.iterdir()):
        if not path.is_dir():
            continue
        match = re.search(r"(\d+)$", path.name)
        if match is None:
            continue
        obj_id = int(match.group(1))
        has_masks = any(p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in path.iterdir())
        if has_masks:
            mapping[obj_id] = path
    return mapping


class SamBody4DMaskletExtractor:
    """Run external SAM-Body4D command and collect masklets."""

    def __init__(self, command: str) -> None:
        self.command = command

    def extract(
        self,
        video_path: Path,
        frames: list[np.ndarray],
        output_dir: Path,
        config: PreprocessConfig,
    ) -> SamBody4DExtraction:
        if not self.command:
            raise ValueError("sam_body4d.command is empty.")

        output_dir.mkdir(parents=True, exist_ok=True)
        sample_dir = output_dir.parent
        requested_mhr_path = sample_dir / "mhr_sequence.jsonl"
        requested_sam3_first_mask = sample_dir / ".tmp" / "sam3_first_mask.png"
        requested_subject_root = sample_dir / "sam3_subject_masks"
        requested_subjects_json = sample_dir / "sam3_subjects.json"
        command = self.command.format(
            video=str(video_path),
            output_dir=str(output_dir),
            masklet_dir=str(output_dir),
            sample_dir=str(sample_dir),
            ckpt_root=str(Path(config.ckpt_root).resolve()),
            prompt=config.mask_prompt,
            mask_prompt=config.mask_prompt,
            output_mhr=str(requested_mhr_path),
            mhr_output=str(requested_mhr_path),
            sam3_first_mask=str(requested_sam3_first_mask),
            first_mask_output=str(requested_sam3_first_mask),
            sam3_subject_root=str(requested_subject_root),
            subject_mask_root=str(requested_subject_root),
            sam3_subjects_json=str(requested_subjects_json),
            subjects_json=str(requested_subjects_json),
        )
        project_root = Path(__file__).resolve().parents[2]
        subprocess.run(command, shell=True, check=True, cwd=str(project_root))

        direct_masks = [p for p in output_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        nested_mask_dir = output_dir / "masks"
        nested_masks = []
        if nested_mask_dir.exists():
            nested_masks = [p for p in nested_mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

        if direct_masks:
            masks = _load_mask_images(output_dir)
        elif nested_masks:
            masks = _load_mask_images(nested_mask_dir)
            _save_masks(masks, output_dir)
        else:
            npz_path = output_dir / "masklets.npz"
            if not npz_path.exists():
                raise RuntimeError(
                    "SAM-Body4D finished but produced neither image masks nor masklets.npz in output_dir."
                )
            loaded = np.load(npz_path)
            if len(loaded.files) == 0:
                raise RuntimeError(f"Empty masklets npz: {npz_path}")
            masks = loaded[loaded.files[0]].astype(bool)
            if masks.ndim != 3:
                raise RuntimeError(f"masklets npz must be T,H,W, got shape {masks.shape}")
            _save_masks(masks, output_dir)

        if len(frames) > 0:
            t = min(len(frames), masks.shape[0])
            masks = masks[:t]

        mhr_source = requested_mhr_path if requested_mhr_path.exists() else _discover_mhr_file(sample_dir)
        if mhr_source is None:
            raise RuntimeError(
                "SAM-Body4D finished but MHR sequence file was not found. "
                "Expected `{output_mhr}` or a file with `mhr/skeleton/rig` in name."
            )

        mhr_target = sample_dir / f"mhr_sequence{mhr_source.suffix.lower()}"
        if mhr_source.resolve() != mhr_target.resolve():
            shutil.copy2(mhr_source, mhr_target)

        first_mask_path = requested_sam3_first_mask if requested_sam3_first_mask.exists() else None
        subject_mask_dirs = _discover_subject_mask_dirs(requested_subject_root)
        subject_manifest_path = requested_subjects_json if requested_subjects_json.exists() else None
        return SamBody4DExtraction(
            masks=masks.astype(bool),
            mhr_path=mhr_target,
            sam3_first_mask_path=first_mask_path,
            subject_mask_dirs=subject_mask_dirs,
            subject_manifest_path=subject_manifest_path,
        )
