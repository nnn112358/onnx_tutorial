#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-hajimete_no_onnx.md}
OUTDIR=${2:-build}
mkdir -p "$OUTDIR"

if command -v pandoc >/dev/null 2>&1; then
  echo "[build] Generating PDF and HTML from $SOURCE"
  pandoc "$SOURCE" -o "$OUTDIR/$(basename "${SOURCE%.md}").pdf"
  pandoc "$SOURCE" -o "$OUTDIR/$(basename "${SOURCE%.md}").html" --standalone
else
  echo "[warn] pandoc not found. Install pandoc to enable PDF/HTML export." >&2
  exit 1
fi
