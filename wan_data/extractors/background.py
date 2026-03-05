from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


class OmniEraserBackgroundExtractor:
    """Run external OmniEraser command to generate clean background."""

    def __init__(self, command: str) -> None:
        self.command = command

    def extract(
        self,
        video_path: Path,
        frames: list[np.ndarray],
        masks: np.ndarray,
        masklet_dir: Path,
        output_background: Path,
    ) -> np.ndarray:
        if not self.command:
            raise ValueError("omni_eraser.command is empty.")
        if not frames:
            raise ValueError("No frames available for OmniEraser background extraction.")
        if masks.ndim != 3 or masks.shape[0] == 0:
            raise ValueError(f"Invalid masks for OmniEraser extraction, expected T,H,W and T>0, got {masks.shape}")

        output_background.parent.mkdir(parents=True, exist_ok=True)
        omni_input_dir = output_background.parent / "omni_inputs"
        omni_input_dir.mkdir(parents=True, exist_ok=True)
        first_frame_path = omni_input_dir / "first_frame.png"
        first_mask_path = omni_input_dir / "first_mask.png"

        Image.fromarray(frames[0].astype(np.uint8), mode="RGB").save(first_frame_path)
        Image.fromarray((masks[0].astype(np.uint8) * 255), mode="L").save(first_mask_path)

        command = self.command.format(
            video=str(video_path),
            masklet_dir=str(masklet_dir),
            output_background=str(output_background),
            first_frame=str(first_frame_path),
            first_mask=str(first_mask_path),
            frame0=str(first_frame_path),
            mask0=str(first_mask_path),
        )
        subprocess.run(command, shell=True, check=True)

        if not output_background.exists():
            raise RuntimeError(f"OmniEraser command finished but output file not found: {output_background}")

        image = np.asarray(Image.open(output_background).convert("RGB"), dtype=np.uint8)
        return image
