# wan-data

`wan-data` is a preprocessing framework for controllable WAN video training data.

It converts each target video into:

- `target_video`
- `subject_portrait` (reference identity image)
- `background` (clean scene)
- `masklets` (temporal human masks)

Skeleton/MHR extraction is intentionally **not included**.

## Output Layout

```text
processed_dataset/
  dataset.jsonl
  samples/
    <sample_id>/
      target_video.mp4
      subject_portrait.png
      background.png
      masklets/
        000000.png
        000001.png
        ...
      metadata.json
```

`dataset.jsonl` contains one JSON record per sample, with relative paths to these artifacts.

## Quick Start

1. Create config:

```bash
cp configs/preprocess.example.yaml configs/preprocess.yaml
```

2. Put videos under `videos/` (or set `input_path` in config).

3. Run:

```bash
python scripts/preprocess.py --config configs/preprocess.yaml
```

Optional:

```bash
python scripts/preprocess.py --config configs/preprocess.yaml --dry-run
python scripts/preprocess.py --config configs/preprocess.yaml --overwrite
```

## Backends

This framework is now strict and only supports:

- `masklet_source: sam_body4d`
  - runs external command template from `sam_body4d.command`.
  - placeholders: `{video}`, `{output_dir}`.
- `background_source: omni_eraser`
  - runs external command template from `omni_eraser.command`.
  - placeholders: `{video}`, `{masklet_dir}`, `{output_background}`, `{first_frame}`, `{first_mask}`.
  - current pipeline uses OmniEraser on the first frame only (`first_frame + first_mask`) to produce `background.png`.

If either command is empty, preprocessing exits with an error.

## Notes

- Video decoding requires at least one available backend:
  - `opencv-python`, or
  - a working `torchvision` video backend (PyAV/FFmpeg).
- You can also pass an image-sequence directory as input path for debugging decoding issues.
