from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import PreprocessConfig


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
    ) -> np.ndarray:
        if not self.command:
            raise ValueError("sam_body4d.command is empty.")

        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.command.format(video=str(video_path), output_dir=str(output_dir))
        subprocess.run(command, shell=True, check=True)

        if any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in output_dir.iterdir()):
            masks = _load_mask_images(output_dir)
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
        return masks.astype(bool)
