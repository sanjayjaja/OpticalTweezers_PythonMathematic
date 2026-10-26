# Optical tweezer model by Sanjay Khadka
# Analysis added by Nico: pixel sums, full-array histogram fits,
# loading/fidelity estimates, bootstrap uncertainty and fit checks.

import argparse
from pathlib import Path
import numpy as np

from models.imageprocessing import ImageProcessing
from models.array_analysis import (
    detect_sites,
    analyse_array,
    bootstrap_sites,
    sensitivity_checks,
    export_results,
)


def main():
    # Read settings from the command line.
    parser = argparse.ArgumentParser(description="S41 pixel-sum loading analysis")
    root = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=root / "S41")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--peak-threshold", type=float, default=550)
    parser.add_argument(
        "--background",
        type=int,
        nargs=4,
        default=(110, 150, 35, 105),
        metavar=("Y0", "Y1", "X0", "X1"),
    )
    parser.add_argument(
        "--positions", type=Path, help="Optional CSV: row,column, with header"
    )
    parser.add_argument(
        "--quick", action="store_true", help="One-site smoke test; 20 resamples"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save plots without opening windows"
    )
    args = parser.parse_args()
    if args.bootstrap < 0 or args.radius < 1 or args.bins < 10 or args.block_size < 1:
        parser.error("Use bootstrap >= 0, radius >= 1, bins >= 10, block-size >= 1.")
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg", force=True)

    # Read original 16-bit pixels and retain the original ten-row crop.
    loader = ImageProcessing()
    images, average = loader.load_tga_folder(args.data)
    if images.shape[2] < 20:
        parser.error("At least 20 images are needed for these histogram fits.")
    # Use supplied positions, or find the spots in the average image.
    if args.positions:
        positions = np.atleast_1d(
            np.genfromtxt(args.positions, delimiter=",", names=True)
        )
        centres = np.column_stack([positions["row"], positions["column"]]).astype(int)
    else:
        centres = detect_sites(average, args.radius, args.peak_threshold)
    if args.quick:
        # Use the same example site as the first one-tweezer version.
        index = np.argmin(np.sum((centres - np.array([44, 50])) ** 2, axis=1))
        centres = centres[index : index + 1]
        args.bootstrap = 20
        args.output = args.output / "quick"
    print(
        f"Loaded {images.shape[2]} images; detected/selected {len(centres)} sites.",
        flush=True,
    )
    # Measure signals and fit a histogram for each tweezer.
    results = analyse_array(
        images, average, centres, args.radius, args.background, args.bins
    )
    fitted = [r for r in results if "signals" in r]
    if not fitted:
        raise RuntimeError("No valid fits. Check positions, ROI sizes and background.")
    # Resample the trials to estimate uncertainty.
    pooled_ci = bootstrap_sites(results, args.bootstrap, args.seed, args.block_size)
    # Repeat a few choices for the original example site.
    representative = min(
        fitted, key=lambda r: (r["row"] - 44) ** 2 + (r["column"] - 50) ** 2
    )
    sensitivity = sensitivity_checks(
        images, representative, tuple(args.background), args.bins
    )
    config = dict(
        method="background-corrected pixel sums; two 1D Gaussian histogram peaks",
        bootstrap=args.bootstrap,
        seed=args.seed,
        block_size=args.block_size,
        radius=args.radius,
        bins=args.bins,
        peak_threshold=args.peak_threshold,
        background=list(args.background),
        crop_rows=10,
        sensitivity_site=representative["site"],
        bootstrap_interval="95% percentile; conditional on model, detected sites and preprocessing",
    )
    # Save the measurements, results and plots.
    summary = export_results(
        args.output,
        images,
        average,
        centres,
        results,
        loader.image_files,
        config,
        pooled_ci,
        sensitivity,
    )
    print(f"Fits completed: {len(fitted)}/{len(results)}")
    print(
        f'Mean classified loading (includes review sites): {summary["pooled_classified_fraction"]:.1%}'
    )
    usable = sum(np.isfinite(r["loading_fraction_ci_low"]) for r in fitted)
    print(f"Sites with usable bootstrap intervals: {usable}/{len(results)}")
    print(
        f'Example site {representative["site"]}: {representative["loading_fraction"]:.1%} classified occupied'
    )
    if np.isfinite(representative["loading_fraction_ci_low"]):
        print(
            f'Example site 95% interval: {representative["loading_fraction_ci_low"]:.1%} to {representative["loading_fraction_ci_high"]:.1%}'
        )
    if pooled_ci:
        print(
            f"95% bootstrap interval for array mean: {pooled_ci[0]:.1%} to {pooled_ci[1]:.1%}"
        )
    print(f'Sites needing review: {summary["review_sites"]}')
    print(f"Results saved in: {args.output.resolve()}")
    print(
        "Fidelity and its intervals assume the fitted Gaussian model, not known atom labels."
    )
    if args.bootstrap < 100:
        print(
            "Smoke-test run: uncertainty intervals suppressed; use at least 100 resamples."
        )
    # Open the overview plots unless --no-show was selected.
    if not args.no_show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        for name in ("detected_sites.png", "array_maps.png", "loading_uncertainty.png"):
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.imshow(mpimg.imread(args.output / name))
            ax.axis("off")
            fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
