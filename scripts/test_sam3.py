#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick SAM3 test: run prompt-based video propagation and report stability metrics."
    )
    parser.add_argument("--video", type=str, default=None, help="Input video path. Required unless --load-only.")
    parser.add_argument("--prompt", type=str, default="person", help="Text prompt injected to SAM3.")
    parser.add_argument("--sam3-repo", type=str, default="third_party/sam3", help="Path to SAM3 repo.")
    parser.add_argument("--checkpoint", type=str, default="ckpts/sam3/sam3.pt", help="SAM3 checkpoint path.")
    parser.add_argument("--bpe", type=str, default="third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz", help="Optional BPE path.")
    parser.add_argument("--frame-index", type=int, default=0, help="Prompt frame index.")
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=["both", "forward", "backward"],
        help="Propagation direction.",
    )
    parser.add_argument("--mask-object-id", type=int, default=0, help="0=merge all objects; >0=single object id.")
    parser.add_argument("--output-dir", type=str, default="tmp/sam3_test", help="Output directory.")
    parser.add_argument("--load-only", action="store_true", help="Only validate model import and checkpoint load.")
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return (REPO_ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def _to_bool_mask(
    out_binary_masks: np.ndarray,
    out_obj_ids: np.ndarray,
    object_id: int,
    height: int,
    width: int,
) -> np.ndarray:
    if out_binary_masks.size == 0:
        return np.zeros((height, width), dtype=bool)
    if object_id > 0:
        keep = out_obj_ids == object_id
        if keep.any():
            return np.any(out_binary_masks[keep], axis=0).astype(bool)
        return np.zeros((height, width), dtype=bool)
    return np.any(out_binary_masks, axis=0).astype(bool)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _compute_metrics(mask_paths: list[Path]) -> dict[str, float]:
    areas: list[float] = []
    for p in mask_paths:
        m = np.asarray(Image.open(p).convert("L"), dtype=np.uint8) > 127
        areas.append(float(m.mean()))
    area_arr = np.array(areas, dtype=np.float64) if areas else np.zeros((0,), dtype=np.float64)

    if area_arr.size == 0:
        return {
            "num_frames": 0.0,
            "non_empty_ratio": 0.0,
            "mean_area_ratio": 0.0,
            "area_flicker": 0.0,
        }

    return {
        "num_frames": float(area_arr.size),
        "non_empty_ratio": float((area_arr > 0.001).mean()),
        "mean_area_ratio": float(area_arr.mean()),
        "area_flicker": float(np.abs(np.diff(area_arr)).mean()) if area_arr.size > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()

    sam3_repo = resolve_repo_path(args.sam3_repo)
    checkpoint = resolve_repo_path(args.checkpoint)
    bpe = resolve_repo_path(args.bpe) if args.bpe else None
    output_dir = resolve_repo_path(args.output_dir)

    if not sam3_repo.exists():
        raise FileNotFoundError(f"SAM3 repo not found: {sam3_repo}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
    if bpe is not None and not bpe.exists():
        raise FileNotFoundError(f"SAM3 BPE file not found: {bpe}")

    sam3_repo_str = str(sam3_repo)
    if sam3_repo_str not in sys.path:
        sys.path.insert(0, sam3_repo_str)

    try:
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    except Exception as exc:
        raise RuntimeError("Failed to import SAM3. Install SAM3 runtime dependencies first.") from exc

    predictor = Sam3VideoPredictor(
        checkpoint_path=str(checkpoint),
        bpe_path=str(bpe) if bpe is not None else None,
        async_loading_frames=False,
        video_loader_type="cv2",
        compile=False,
    )

    if args.load_only:
        print("SAM3 load OK")
        predictor.shutdown()
        return

    if args.video is None:
        predictor.shutdown()
        raise ValueError("--video is required unless --load-only is set.")

    video = resolve_repo_path(args.video)
    if not video.exists():
        predictor.shutdown()
        raise FileNotFoundError(f"Input video not found: {video}")
    if not args.prompt.strip():
        predictor.shutdown()
        raise ValueError("--prompt must be non-empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    masklet_dir = output_dir / "masklets"
    subject_root = output_dir / "subjects"
    manifest_path = output_dir / "subjects.json"
    metrics_path = output_dir / "metrics.json"

    start_resp = predictor.handle_request({"type": "start_session", "resource_path": str(video)})
    session_id = start_resp["session_id"]

    add_resp = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": args.frame_index,
            "text": args.prompt,
        }
    )

    session_state = predictor._ALL_INFERENCE_STATES[session_id]["state"]  # noqa: SLF001
    num_frames = int(session_state["num_frames"])
    height = int(session_state["orig_height"])
    width = int(session_state["orig_width"])

    frame_outputs: dict[int, dict] = {int(add_resp["frame_index"]): add_resp["outputs"]}
    for resp in predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": args.direction,
            "start_frame_index": args.frame_index,
            "max_frame_num_to_track": num_frames,
        }
    ):
        frame_outputs[int(resp["frame_index"])] = resp["outputs"]

    all_obj_ids: set[int] = set()
    for outputs in frame_outputs.values():
        for obj_id in outputs["out_obj_ids"].tolist():
            all_obj_ids.add(int(obj_id))
    sorted_obj_ids = sorted(all_obj_ids)

    for obj_id in sorted_obj_ids:
        (subject_root / f"obj_{obj_id:04d}").mkdir(parents=True, exist_ok=True)

    for idx in range(num_frames):
        outputs = frame_outputs.get(idx, None)
        if outputs is None:
            mask = np.zeros((height, width), dtype=bool)
            frame_obj_mask: dict[int, np.ndarray] = {}
        else:
            out_binary_masks = outputs["out_binary_masks"].astype(bool)
            out_obj_ids = outputs["out_obj_ids"]
            mask = _to_bool_mask(
                out_binary_masks=out_binary_masks,
                out_obj_ids=out_obj_ids,
                object_id=args.mask_object_id,
                height=height,
                width=width,
            )
            frame_obj_mask = {}
            for row, obj_id in enumerate(out_obj_ids.tolist()):
                frame_obj_mask[int(obj_id)] = out_binary_masks[row]

        _save_mask(mask, masklet_dir / f"{idx:06d}.png")
        for obj_id in sorted_obj_ids:
            subject_mask = frame_obj_mask.get(obj_id, np.zeros((height, width), dtype=bool))
            _save_mask(subject_mask, subject_root / f"obj_{obj_id:04d}" / f"{idx:06d}.png")

    manifest = {
        "video": str(video),
        "prompt": args.prompt,
        "frame_index": int(args.frame_index),
        "direction": args.direction,
        "mask_object_id": int(args.mask_object_id),
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "subjects": [
            {"obj_id": obj_id, "mask_dir": str((subject_root / f"obj_{obj_id:04d}").resolve())}
            for obj_id in sorted_obj_ids
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    mask_paths = sorted(masklet_dir.glob("*.png"))
    metrics = _compute_metrics(mask_paths)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    predictor.shutdown()

    print(f"SAM3 run OK. Outputs: {output_dir}")
    print(f"frames={int(metrics['num_frames'])}")
    print(f"non_empty_ratio={metrics['non_empty_ratio']:.4f}")
    print(f"mean_area_ratio={metrics['mean_area_ratio']:.6f}")
    print(f"area_flicker={metrics['area_flicker']:.6f}")

    if metrics["non_empty_ratio"] == 0.0:
        raise RuntimeError("All predicted masks are empty. Check prompt/video/checkpoint.")


if __name__ == "__main__":
    main()
