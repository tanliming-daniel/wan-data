# wan-data

`wan-data` is a preprocessing framework for controllable WAN video training data.

It converts each target video into:

- `target_video`
- `subject_portrait` (merged fallback portrait)
- `subject_portraits` (per-subject portraits with `obj_id`)
- `background` (clean scene)
- `masklets` (temporal human masks)
- `sam3_subject_masks` (per-subject full-video masks from SAM3 prompt propagation)
- `mhr_sequence` (saved MHR sequence file)

## Output Layout

```text
processed_dataset/
  dataset.jsonl
  samples/
    <sample_id>/
      target_video.mp4
      subject_portrait.png
      background.png
      mhr_sequence.jsonl
      sam3_first_mask.png
      sam3_subjects.json
      subject_portraits/
        obj_0001.png
        obj_0002.png
      sam3_subject_masks/
        obj_0001/
          000000.png
          000001.png
        obj_0002/
          000000.png
          000001.png
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
python scripts/preprocess.py --config configs/preprocess.yaml --prompt "person in red shirt"
python scripts/preprocess.py --config configs/preprocess.yaml --ckpt-root ./ckpts
```

## Backends

This framework is now strict and only supports:

- `masklet_source: sam_body4d`
  - runs external command template from `sam_body4d.command`.
  - placeholders: `{video}`, `{output_dir}`, `{masklet_dir}`, `{sample_dir}`, `{ckpt_root}`, `{output_mhr}`, `{sam3_first_mask}`, `{sam3_subject_root}`, `{sam3_subjects_json}`, `{mask_prompt}`.
  - integration workflow:
    - use SAM3 (`third_party/sam3`) text prompt to propagate full-video masks;
    - use SAM-Body4D (`third_party/sam-body4d`) to export `mhr_sequence.jsonl`.
  - set prompt with `mask_prompt` in config or `--prompt` in CLI.
  - pipeline requires MHR output and stores it as `mhr_sequence.*` in sample folder.
  - default integration script: `scripts/integrations/run_sam_body4d.py`.
- `background_source: omni_eraser`
  - runs external command template from `omni_eraser.command`.
  - placeholders: `{video}`, `{masklet_dir}`, `{output_background}`, `{first_frame}`, `{first_mask}`.
  - current pipeline uses OmniEraser on the first frame only (`first_frame + first_mask`) to produce `background.png`.
  - `first_mask` is taken from `sam3_first_mask.png` (SAM3 output) when available.

If either command is empty, preprocessing exits with an error.

Metadata includes multi-subject records:

- `subjects`: list of `{obj_id, portrait, mask_dir}`
- `ref_portrait`: defaults to first subject portrait when available

## Checkpoints

Use a unified root directory: `ckpts/`.

- `ckpts/sam3/sam3.pt`
- `ckpts/sam-3d-body-dinov3/model.ckpt`
- `ckpts/sam-3d-body-dinov3/model_config.yaml`
- `ckpts/sam-3d-body-dinov3/assets/mhr_model.pt`
- `ckpts/moge-2-vitl-normal/model.pt`
- `ckpts/diffusion-vas-amodal-segmentation/`
- `ckpts/diffusion-vas-content-completion/`
- `ckpts/depth_anything_v2_vitl.pth`

Download all required checkpoints:

```bash
python scripts/download_ckpts.py --ckpt-root ./ckpts
```

For gated models (SAM3 / SAM-3D-Body), provide Hugging Face token:

```bash
HF_TOKEN=hf_xxx python scripts/download_ckpts.py
```

If you want best-effort download and keep exit code 0 on partial failures:

```bash
python scripts/download_ckpts.py --allow-partial
```

`third_party/sam-body4d/configs/body4d.yaml` is updated at runtime using your configured `ckpt_root` (written as a relative path).
You can override at pipeline level with config key `ckpt_root` or CLI `--ckpt-root`.

## Notes

- SAM3 prompt workflow needs accessible SAM3 checkpoints (e.g. HF auth/login if using default download).
- Video decoding requires at least one available backend:
  - `opencv-python`, or
  - a working `torchvision` video backend (PyAV/FFmpeg).
- You can also pass an image-sequence directory as input path for debugging decoding issues.
