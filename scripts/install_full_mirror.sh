#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EXTRA="${1:-full}"
MIRROR_BASE="${GITHUB_MIRROR_BASE:-https://githubfast.com/}"
PIP_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

echo "[INFO] repo root: ${ROOT_DIR}"
echo "[INFO] install extra: ${EXTRA}"
echo "[INFO] github mirror: ${MIRROR_BASE}"

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git not found"
  exit 1
fi

if ! command -v pip >/dev/null 2>&1; then
  echo "[ERROR] pip not found in current environment"
  exit 1
fi

echo "[STEP] init submodules"
git submodule update --init --recursive

echo "[STEP] upgrade pip toolchain"
pip install -U pip wheel "setuptools<81"

echo "[STEP] install project with mirror rewrite"
GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=url."${MIRROR_BASE}".insteadOf \
GIT_CONFIG_VALUE_0=https://github.com/ \
PIP_DEFAULT_TIMEOUT="${PIP_TIMEOUT}" \
pip install -e ".[${EXTRA}]"

echo "[DONE] install finished"
