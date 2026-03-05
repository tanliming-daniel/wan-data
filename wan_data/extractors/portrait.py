from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..io.image_ops import expand_bbox, mask_bbox


class SubjectPortraitExtractor:
    def __init__(self, margin: float = 0.2) -> None:
        self.margin = max(0.0, float(margin))

    def extract(self, frames: list[np.ndarray], masks: np.ndarray, output_path: Path) -> np.ndarray:
        if not frames:
            raise RuntimeError("No frames available for portrait extraction.")

        frame_stack = np.stack(frames, axis=0)
        if masks.ndim != 3:
            raise RuntimeError(f"Masks must be T,H,W. Got: {masks.shape}")
        t = min(frame_stack.shape[0], masks.shape[0])
        frame_stack = frame_stack[:t]
        masks = masks[:t]

        areas = masks.reshape(t, -1).sum(axis=1)
        best_idx = int(np.argmax(areas)) if areas.size > 0 else 0
        frame = frame_stack[best_idx]
        mask = masks[best_idx]
        h, w = frame.shape[:2]

        bbox = mask_bbox(mask)
        if bbox is None:
            side = int(min(h, w) * 0.6)
            x0 = max(0, (w - side) // 2)
            y0 = max(0, (h - side) // 2)
            x1, y1 = x0 + side, y0 + side
        else:
            x0, y0, x1, y1 = expand_bbox(bbox, height=h, width=w, margin_ratio=self.margin)

        crop = frame[y0:y1, x0:x1]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop, mode="RGB").save(output_path)
        return crop
