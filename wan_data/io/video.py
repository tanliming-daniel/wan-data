from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _load_image_sequence(input_dir: Path, max_frames: int | None = None) -> list[np.ndarray]:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]
    if not files:
        raise RuntimeError(f"No image frames found in directory: {input_dir}")
    if max_frames is not None:
        files = files[:max_frames]

    frames: list[np.ndarray] = []
    for path in files:
        image = Image.open(path).convert("RGB")
        frames.append(np.asarray(image, dtype=np.uint8))
    return frames


def _load_video_with_opencv(video_path: Path, max_frames: int | None = None) -> list[np.ndarray] | None:
    try:
        import cv2
    except Exception:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.uint8, copy=False))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _load_video_with_torchvision(video_path: Path, max_frames: int | None = None) -> list[np.ndarray] | None:
    try:
        from torchvision.io import read_video
    except Exception:
        return None

    try:
        video, _, _ = read_video(str(video_path), pts_unit="sec")
    except Exception:
        return None

    if video.ndim != 4:
        return None
    if max_frames is not None:
        video = video[:max_frames]
    return [frame.numpy().astype(np.uint8, copy=False) for frame in video]


def read_frames(input_path: str | Path, max_frames: int | None = None) -> list[np.ndarray]:
    """Load RGB uint8 frames from a video file or image-sequence directory."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_dir():
        return _load_image_sequence(path, max_frames=max_frames)

    frames = _load_video_with_opencv(path, max_frames=max_frames)
    if frames:
        return frames

    frames = _load_video_with_torchvision(path, max_frames=max_frames)
    if frames:
        return frames

    raise RuntimeError(
        "Unable to decode video. Install one of: "
        "`opencv-python` or a working `torchvision` video backend (PyAV/FFmpeg)."
    )
