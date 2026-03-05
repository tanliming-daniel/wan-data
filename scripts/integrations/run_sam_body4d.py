from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM3 text prompt for full-video masklets, then run SAM-Body4D for MHR."
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output masklet directory (PNG sequence).",
    )
    parser.add_argument(
        "--output_mhr",
        type=str,
        required=True,
        help="Output MHR sequence file path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for SAM3 (e.g. 'person', 'a woman', 'a dancer').",
    )
    parser.add_argument(
        "--sam3_repo",
        type=str,
        default="third_party/sam3",
        help="SAM3 repo root.",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="ckpts",
        help="Unified checkpoints root for SAM3 and SAM-Body4D.",
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=None,
        help="SAM3 checkpoint path. Default: <ckpt_root>/sam3/sam3.pt.",
    )
    parser.add_argument(
        "--sam3_bpe",
        type=str,
        default=None,
        help="Optional SAM3 BPE path.",
    )
    parser.add_argument(
        "--sam3_frame_index",
        type=int,
        default=0,
        help="Frame index where text prompt is injected.",
    )
    parser.add_argument(
        "--sam3_direction",
        type=str,
        default="both",
        choices=["both", "forward", "backward"],
        help="Propagation direction for SAM3 video predictor.",
    )
    parser.add_argument(
        "--sam_repo",
        type=str,
        default="third_party/sam-body4d",
        help="SAM-Body4D repo root.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Raw SAM-Body4D output dir.",
    )
    parser.add_argument(
        "--sam3_first_mask",
        type=str,
        default=None,
        help="Output path for first-frame SAM3 mask.",
    )
    parser.add_argument(
        "--sam3_subject_root",
        type=str,
        default=None,
        help="Output root dir for per-subject SAM3 masks.",
    )
    parser.add_argument(
        "--sam3_subjects_json",
        type=str,
        default=None,
        help="Output JSON manifest path for per-subject masks.",
    )
    parser.add_argument(
        "--mask_object_id",
        type=int,
        default=0,
        help="Object id to export. 0 means merge all predicted objects.",
    )
    return parser.parse_args()


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


def run_sam3_prompt_masks(
    video: Path,
    output_dir: Path,
    first_mask_out: Path,
    subject_root: Path,
    subjects_json: Path,
    prompt: str,
    sam3_repo: Path,
    sam3_checkpoint: Path | None,
    sam3_bpe: Path | None,
    frame_index: int,
    direction: str,
    object_id: int,
) -> None:
    if not sam3_repo.exists():
        raise FileNotFoundError(f"SAM3 repo not found: {sam3_repo}")
    if not prompt.strip():
        raise ValueError("SAM3 prompt is empty.")

    sam3_repo_str = str(sam3_repo.resolve())
    if sam3_repo_str not in sys.path:
        sys.path.insert(0, sam3_repo_str)

    try:
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    except Exception as e:
        raise RuntimeError(
            "Failed to import SAM3. Ensure dependencies are installed in current environment."
        ) from e

    predictor = Sam3VideoPredictor(
        checkpoint_path=str(sam3_checkpoint) if sam3_checkpoint is not None else None,
        bpe_path=str(sam3_bpe) if sam3_bpe is not None else None,
        async_loading_frames=False,
        video_loader_type="cv2",
        compile=False,
    )

    start_resp = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(video),
        }
    )
    session_id = start_resp["session_id"]

    add_resp = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": prompt,
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
            "propagation_direction": direction,
            "start_frame_index": frame_index,
            "max_frame_num_to_track": num_frames,
        }
    ):
        frame_outputs[int(resp["frame_index"])] = resp["outputs"]

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_root.mkdir(parents=True, exist_ok=True)

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
                object_id=object_id,
                height=height,
                width=width,
            )
            frame_obj_mask = {}
            for row, obj_id in enumerate(out_obj_ids.tolist()):
                frame_obj_mask[int(obj_id)] = out_binary_masks[row]

        _save_mask(mask, output_dir / f"{idx:06d}.png")
        if idx == frame_index:
            _save_mask(mask, first_mask_out)

        for obj_id in sorted_obj_ids:
            subject_mask = frame_obj_mask.get(
                obj_id,
                np.zeros((height, width), dtype=bool),
            )
            _save_mask(
                subject_mask,
                subject_root / f"obj_{obj_id:04d}" / f"{idx:06d}.png",
            )

    subjects_manifest = {
        "prompt": prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "subjects": [
            {
                "obj_id": obj_id,
                "mask_dir": str((subject_root / f"obj_{obj_id:04d}").resolve()),
            }
            for obj_id in sorted_obj_ids
        ],
    }
    subjects_json.parent.mkdir(parents=True, exist_ok=True)
    subjects_json.write_text(
        json.dumps(subjects_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    predictor.shutdown()


def run_sam_body4d_for_mhr(video: Path, sam_repo: Path, work_dir: Path) -> Path:
    if not sam_repo.exists():
        raise FileNotFoundError(f"SAM-Body4D repo not found: {sam_repo}")
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(sam_repo / "scripts" / "offline_app.py"),
        "--input_video",
        str(video),
        "--output_dir",
        str(work_dir),
    ]
    subprocess.run(command, check=True, cwd=str(sam_repo))
    mhr_src = work_dir / "mhr_sequence.jsonl"
    if not mhr_src.exists():
        raise RuntimeError(f"SAM-Body4D did not produce MHR sequence: {mhr_src}")
    return mhr_src


def ensure_body4d_ckpt_root(sam_repo: Path, ckpt_root: Path) -> None:
    cfg_path = sam_repo / "configs" / "body4d.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"SAM-Body4D config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if "paths" not in data or not isinstance(data["paths"], dict):
        data["paths"] = {}
    ckpt_value = os.path.relpath(ckpt_root.resolve(), start=sam_repo.resolve())
    ckpt_value = ckpt_value.replace("\\", "/")
    if data["paths"].get("ckpt_root") == ckpt_value:
        return
    data["paths"]["ckpt_root"] = ckpt_value
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    video = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_mhr = Path(args.output_mhr).expanduser().resolve()
    sam_repo = Path(args.sam_repo).expanduser().resolve()
    sam3_repo = Path(args.sam3_repo).expanduser().resolve()
    ckpt_root = Path(args.ckpt_root).expanduser().resolve()
    sam3_checkpoint = (
        Path(args.sam3_checkpoint).expanduser().resolve()
        if args.sam3_checkpoint is not None
        else (ckpt_root / "sam3" / "sam3.pt")
    )
    sam3_bpe = (
        Path(args.sam3_bpe).expanduser().resolve()
        if args.sam3_bpe is not None
        else None
    )
    if args.work_dir is not None:
        work_dir = Path(args.work_dir).expanduser().resolve()
    else:
        work_dir = output_dir.parent / "sam_body4d_raw"
    if args.sam3_first_mask is not None:
        first_mask_out = Path(args.sam3_first_mask).expanduser().resolve()
    else:
        first_mask_out = output_dir.parent / "sam3_first_mask.png"
    if args.sam3_subject_root is not None:
        subject_root = Path(args.sam3_subject_root).expanduser().resolve()
    else:
        subject_root = output_dir.parent / "sam3_subject_masks"
    if args.sam3_subjects_json is not None:
        subjects_json = Path(args.sam3_subjects_json).expanduser().resolve()
    else:
        subjects_json = output_dir.parent / "sam3_subjects.json"

    if not video.exists():
        raise FileNotFoundError(f"Input video not found: {video}")
    if not sam3_checkpoint.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {sam3_checkpoint}")
    if sam3_bpe is not None and not sam3_bpe.exists():
        raise FileNotFoundError(f"SAM3 BPE file not found: {sam3_bpe}")
    ensure_body4d_ckpt_root(sam_repo=sam_repo, ckpt_root=ckpt_root)

    run_sam3_prompt_masks(
        video=video,
        output_dir=output_dir,
        first_mask_out=first_mask_out,
        subject_root=subject_root,
        subjects_json=subjects_json,
        prompt=args.prompt,
        sam3_repo=sam3_repo,
        sam3_checkpoint=sam3_checkpoint,
        sam3_bpe=sam3_bpe,
        frame_index=args.sam3_frame_index,
        direction=args.sam3_direction,
        object_id=args.mask_object_id,
    )

    mhr_src = run_sam_body4d_for_mhr(video=video, sam_repo=sam_repo, work_dir=work_dir)
    output_mhr.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mhr_src, output_mhr)


if __name__ == "__main__":
    main()
