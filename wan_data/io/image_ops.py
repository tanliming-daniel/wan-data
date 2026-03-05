from __future__ import annotations

import numpy as np


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0, y0, x1, y1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    margin_ratio: float = 0.2,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))

    nx0 = max(0, x0 - mx)
    ny0 = max(0, y0 - my)
    nx1 = min(width, x1 + mx)
    ny1 = min(height, y1 + my)
    return nx0, ny0, nx1, ny1
