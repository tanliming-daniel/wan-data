# Unified Checkpoints Root

This directory is the unified checkpoints root for:

- `third_party/sam3`
- `third_party/sam-body4d`

Expected layout:

```text
ckpts/
  sam3/
    sam3.pt
  sam-3d-body-dinov3/
    model.ckpt
    model_config.yaml
    assets/
      mhr_model.pt
  moge-2-vitl-normal/
    model.pt
  diffusion-vas-amodal-segmentation/
  diffusion-vas-content-completion/
  depth_anything_v2_vitl.pth
```

`third_party/sam-body4d/configs/body4d.yaml` is updated at runtime to use this directory via:

`paths.ckpt_root: ../../ckpts` (relative to `third_party/sam-body4d`)

Use the unified downloader:

`python scripts/download_ckpts.py --ckpt-root ./ckpts`
