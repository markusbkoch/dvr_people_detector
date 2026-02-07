from __future__ import annotations

"""Batch-denoise snapshot images for side-by-side testing."""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Denoise snapshot images into a test output directory")
    parser.add_argument("--input-dir", default="data/snapshots", help="Input directory with source snapshots")
    parser.add_argument(
        "--output-dir",
        default="data/snapshots_denoised_test",
        help="Output directory for denoised snapshots",
    )
    parser.add_argument("--h-luma", type=float, default=7.0, help="Luma filter strength")
    parser.add_argument("--h-color", type=float, default=7.0, help="Color filter strength")
    parser.add_argument(
        "--template-window-size",
        type=int,
        default=7,
        help="Template patch size used by non-local means",
    )
    parser.add_argument(
        "--search-window-size",
        type=int,
        default=21,
        help="Search window size used by non-local means",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Process at most N images (-1 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def denoise_image(
    frame: np.ndarray,
    h_luma: float,
    h_color: float,
    template_window_size: int,
    search_window_size: int,
) -> np.ndarray:
    """Apply non-local means color denoising."""
    return cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        h=h_luma,
        hColor=h_color,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )


def main() -> None:
    """Run denoising job and print summary metrics."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )
    if args.limit > 0:
        files = files[: args.limit]

    processed = 0
    skipped_existing = 0
    failed = 0
    total_mean_abs_diff = 0.0
    total_changed_ratio = 0.0

    for path in files:
        target = output_dir / path.name
        if target.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        src = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if src is None:
            failed += 1
            continue

        denoised = denoise_image(
            src,
            h_luma=args.h_luma,
            h_color=args.h_color,
            template_window_size=args.template_window_size,
            search_window_size=args.search_window_size,
        )

        if not cv2.imwrite(str(target), denoised):
            failed += 1
            continue

        diff = cv2.absdiff(src, denoised)
        total_mean_abs_diff += float(np.mean(diff))
        changed = np.any(diff > 0, axis=2)
        total_changed_ratio += float(np.mean(changed))
        processed += 1

    avg_mean_abs_diff = (total_mean_abs_diff / processed) if processed else 0.0
    avg_changed_ratio = (total_changed_ratio / processed) if processed else 0.0

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Requested files: {len(files)}")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed: {failed}")
    print(f"Average mean abs pixel diff: {avg_mean_abs_diff:.4f}")
    print(f"Average changed-pixel ratio: {avg_changed_ratio:.4f}")


if __name__ == "__main__":
    main()
