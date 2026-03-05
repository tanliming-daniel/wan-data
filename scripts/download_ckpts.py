#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class URLFile:
    name: str
    url: str
    rel_out: str


@dataclass(frozen=True)
class HFFile:
    name: str
    repo_id: str
    filename: str
    rel_out: str
    gated: bool = False
    revision: str = "main"


@dataclass(frozen=True)
class HFRepoDir:
    name: str
    repo_id: str
    rel_out_dir: str
    gated: bool = False
    revision: str = "main"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_ok(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def complete_marker(path: Path) -> Path:
    return path / ".complete"


def dir_complete(path: Path) -> bool:
    if not path.exists() or not path.is_dir() or not any(path.iterdir()):
        return False
    return complete_marker(path).exists()


def get_hf_token(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        token = explicit.strip()
        return token or None

    env_token = os.getenv("HF_TOKEN")
    if env_token:
        token = env_token.strip()
        return token or None

    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def get_hf_hub_funcs():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: huggingface_hub. Install with `pip install huggingface_hub`."
        ) from exc
    return hf_hub_download, snapshot_download, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


def download_url_atomic(item: URLFile, ckpt_root: Path, dry_run: bool) -> bool:
    out_path = ckpt_root / item.rel_out
    ensure_dir(out_path.parent)

    if file_ok(out_path):
        print(f"[SKIP] {item.name}")
        return True

    if dry_run:
        print(f"[DRY ] {item.name} -> {out_path}")
        return True

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    try:
        print(f"[DL  ] {item.name}")
        with urllib.request.urlopen(item.url) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        if not file_ok(tmp):
            raise RuntimeError(f"Downloaded empty file: {tmp}")
        tmp.replace(out_path)
        print(f"[OK  ] {item.name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {item.name}: {exc}")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False


def hf_download_file(item: HFFile, ckpt_root: Path, token: Optional[str], dry_run: bool) -> bool:
    try:
        hf_hub_download, _, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError = get_hf_hub_funcs()
    except ModuleNotFoundError as exc:
        print(f"[FAIL] {item.name}: {exc}")
        return False

    out_path = ckpt_root / item.rel_out
    ensure_dir(out_path.parent)

    if file_ok(out_path):
        print(f"[SKIP] {item.name}")
        return True

    if item.gated and not token:
        print(f"[FAIL] {item.name}: missing HF token (use --hf-token or HF_TOKEN)")
        return False

    if dry_run:
        print(f"[DRY ] {item.name} -> {out_path}")
        return True

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    try:
        print(f"[DL  ] {item.name}")
        cached_path = hf_hub_download(
            repo_id=item.repo_id,
            filename=item.filename,
            revision=item.revision,
            token=token,
            resume_download=True,
        )
        shutil.copy2(cached_path, tmp)
        if not file_ok(tmp):
            raise RuntimeError(f"Downloaded empty file: {tmp}")
        tmp.replace(out_path)
        print(f"[OK  ] {item.name}")
        return True
    except GatedRepoError:
        print(f"[FAIL] {item.name}: gated repo access not granted for current token")
        return False
    except (RepositoryNotFoundError, HfHubHTTPError, OSError, RuntimeError) as exc:
        print(f"[FAIL] {item.name}: {exc}")
        return False


def hf_download_repo_dir(item: HFRepoDir, ckpt_root: Path, token: Optional[str], dry_run: bool) -> bool:
    try:
        _, snapshot_download, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError = get_hf_hub_funcs()
    except ModuleNotFoundError as exc:
        print(f"[FAIL] {item.name}: {exc}")
        return False

    out_dir = ckpt_root / item.rel_out_dir
    ensure_dir(out_dir)

    if dir_complete(out_dir):
        print(f"[SKIP] {item.name}")
        return True

    if item.gated and not token:
        print(f"[FAIL] {item.name}: missing HF token (use --hf-token or HF_TOKEN)")
        return False

    if dry_run:
        print(f"[DRY ] {item.name} -> {out_dir}")
        return True

    try:
        print(f"[DL  ] {item.name}")
        snapshot_download(
            repo_id=item.repo_id,
            revision=item.revision,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
        complete_marker(out_dir).write_text("ok\n", encoding="utf-8")
        print(f"[OK  ] {item.name}")
        return True
    except GatedRepoError:
        print(f"[FAIL] {item.name}: gated repo access not granted for current token")
        return False
    except (RepositoryNotFoundError, HfHubHTTPError, OSError) as exc:
        print(f"[FAIL] {item.name}: {exc}")
        return False


def build_specs() -> tuple[list[URLFile], list[HFRepoDir], list[HFFile]]:
    url_files = [
        URLFile(
            name="Depth Anything v2 (vitl)",
            url="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
            rel_out="depth_anything_v2_vitl.pth",
        ),
    ]

    repo_dirs = [
        HFRepoDir(
            name="Diffusion-VAS amodal segmentation",
            repo_id="kaihuac/diffusion-vas-amodal-segmentation",
            rel_out_dir="diffusion-vas-amodal-segmentation",
            gated=False,
        ),
        HFRepoDir(
            name="Diffusion-VAS content completion",
            repo_id="kaihuac/diffusion-vas-content-completion",
            rel_out_dir="diffusion-vas-content-completion",
            gated=False,
        ),
    ]

    hf_files = [
        HFFile(
            name="MoGe-2 ViTL Normal",
            repo_id="Ruicheng/moge-2-vitl-normal",
            filename="model.pt",
            rel_out="moge-2-vitl-normal/model.pt",
            gated=False,
        ),
        HFFile(
            name="SAM3",
            repo_id="facebook/sam3",
            filename="sam3.pt",
            rel_out="sam3/sam3.pt",
            gated=True,
        ),
        HFFile(
            name="SAM-3D-Body (model.ckpt)",
            repo_id="facebook/sam-3d-body-dinov3",
            filename="model.ckpt",
            rel_out="sam-3d-body-dinov3/model.ckpt",
            gated=True,
        ),
        HFFile(
            name="SAM-3D-Body (model_config.yaml)",
            repo_id="facebook/sam-3d-body-dinov3",
            filename="model_config.yaml",
            rel_out="sam-3d-body-dinov3/model_config.yaml",
            gated=True,
        ),
        HFFile(
            name="SAM-3D-Body (mhr_model.pt)",
            repo_id="facebook/sam-3d-body-dinov3",
            filename="assets/mhr_model.pt",
            rel_out="sam-3d-body-dinov3/assets/mhr_model.pt",
            gated=True,
        ),
    ]

    return url_files, repo_dirs, hf_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download unified checkpoints for SAM3 and SAM-Body4D into ckpts/."
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default=None,
        help="Checkpoint root (default: <repo>/ckpts).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted, use HF_TOKEN or cached login token.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing files.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit with code 0 even if some downloads fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ckpt_root = (
        Path(args.ckpt_root).expanduser().resolve()
        if args.ckpt_root
        else (repo_root / "ckpts").resolve()
    )
    ensure_dir(ckpt_root)

    token = get_hf_token(args.hf_token)
    url_files, repo_dirs, hf_files = build_specs()

    failures = 0
    total = 0

    for item in url_files:
        total += 1
        if not download_url_atomic(item=item, ckpt_root=ckpt_root, dry_run=args.dry_run):
            failures += 1

    for item in repo_dirs:
        total += 1
        if not hf_download_repo_dir(item=item, ckpt_root=ckpt_root, token=token, dry_run=args.dry_run):
            failures += 1

    for item in hf_files:
        total += 1
        if not hf_download_file(item=item, ckpt_root=ckpt_root, token=token, dry_run=args.dry_run):
            failures += 1

    success = total - failures
    print(f"Done: {success}/{total} succeeded. ckpt_root={ckpt_root}")

    if failures > 0 and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
