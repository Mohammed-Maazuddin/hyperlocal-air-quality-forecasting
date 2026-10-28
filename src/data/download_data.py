"""
src/data/download_data.py
--------------------------
Downloads the raw TSPCB Hyderabad IITH Kandi air-quality CSV from the
OpenCity India portal and saves it under data/raw/.

Usage:
    python -m src.data.download_data
"""

from __future__ import annotations

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

import yaml

# ── Locate project root (two levels above this file: src/data → src → project) ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def download_csv(url: str, dest: Path, chunk_size: int = 1 << 16) -> None:
    """Download *url* to *dest*, streaming in chunks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_data] Fetching:\n  {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"HTTP {resp.status} received. Cannot download dataset."
                )
            content_length = resp.headers.get("Content-Length")
            total = int(content_length) if content_length else None

            downloaded = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded:,} / {total:,} bytes  ({pct:.1f}%)", end="")

        print(f"\n[download_data] Saved to: {dest}")
        print(f"[download_data] File size: {dest.stat().st_size:,} bytes")

    except urllib.error.URLError as exc:
        print(f"\n[download_data] ERROR – Could not reach URL: {exc}", file=sys.stderr)
        print(
            "  Check your internet connection or update configs/config.yaml → data.source_url",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]
    filename = cfg["data"]["raw_filename"]
    url = cfg["data"]["source_url"]

    dest = raw_dir / filename

    if dest.exists():
        size = dest.stat().st_size
        print(f"[download_data] File already exists ({size:,} bytes): {dest}")
        answer = input("  Re-download? [y/N] ").strip().lower()
        if answer != "y":
            print("[download_data] Skipping download.")
            return

    download_csv(url, dest)

    # ── Print data source attribution ──────────────────────────────────────────
    print("\n── Source Attribution ─────────────────────────────────────────────")
    print(f"  Portal   : {cfg['data']['source_attribution']}")
    print(f"  License  : {cfg['data']['source_license']}")
    print(f"  URL      : {url}")
    print("────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
