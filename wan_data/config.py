from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SamBody4DConfig:
    enabled: bool = False
    command: str = ""


@dataclass
class OmniEraserConfig:
    enabled: bool = False
    command: str = ""


@dataclass
class PreprocessConfig:
    input_path: str = "input_videos"
    output_root: str = "processed_dataset"
    ckpt_root: str = "ckpts"
    recursive: bool = True
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    max_frames: int | None = None
    copy_target_video: bool = True
    overwrite: bool = False
    dry_run: bool = False

    masklet_source: str = "sam_body4d"
    background_source: str = "omni_eraser"
    mask_prompt: str = "person"
    portrait_margin: float = 0.2

    sam_body4d: SamBody4DConfig = field(default_factory=SamBody4DConfig)
    omni_eraser: OmniEraserConfig = field(default_factory=OmniEraserConfig)

    def normalize(self) -> None:
        self.video_extensions = tuple(
            ext if ext.startswith(".") else f".{ext}" for ext in (e.lower() for e in self.video_extensions)
        )
        if self.max_frames is not None and self.max_frames <= 0:
            self.max_frames = None
        if self.portrait_margin < 0:
            self.portrait_margin = 0.0
        self.mask_prompt = self.mask_prompt.strip()
        if not self.mask_prompt:
            raise ValueError("`mask_prompt` must be a non-empty string.")
        if self.masklet_source != "sam_body4d":
            raise ValueError("Only `sam_body4d` is supported for masklet_source.")
        if self.background_source != "omni_eraser":
            raise ValueError("Only `omni_eraser` is supported for background_source.")


def _load_raw_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain an object/dict: {path}")
    return data


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def load_preprocess_config(config_path: str | Path) -> PreprocessConfig:
    path = Path(config_path).expanduser().resolve()
    data = _load_raw_config(path)

    sam = SamBody4DConfig(**_as_dict(data.get("sam_body4d")))
    omni = OmniEraserConfig(**_as_dict(data.get("omni_eraser")))

    known = {
        "input_path",
        "output_root",
        "ckpt_root",
        "recursive",
        "video_extensions",
        "max_frames",
        "copy_target_video",
        "overwrite",
        "dry_run",
        "masklet_source",
        "background_source",
        "mask_prompt",
        "portrait_margin",
    }
    cfg_kwargs = {k: v for k, v in data.items() if k in known}
    cfg = PreprocessConfig(**cfg_kwargs, sam_body4d=sam, omni_eraser=omni)

    # Resolve relative paths against config directory.
    base = path.parent
    if not Path(cfg.input_path).is_absolute():
        cfg.input_path = str((base / cfg.input_path).resolve())
    if not Path(cfg.output_root).is_absolute():
        cfg.output_root = str((base / cfg.output_root).resolve())
    if not Path(cfg.ckpt_root).is_absolute():
        cfg.ckpt_root = str((base / cfg.ckpt_root).resolve())

    cfg.normalize()
    return cfg
