"""Detect sites, analyse the full array, bootstrap errors, and export results."""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import norm

from .loading_analysis import measure_signals, fit_signals, double_gaussian


def detect_sites(average, radius=3, peak_threshold=550, min_distance=7):
    # Smooth only for locating spots; measurements always use original pixels.
    smoothed = gaussian_filter(average, sigma=1)
    peaks = (smoothed == maximum_filter(smoothed, size=min_distance)) & (
        smoothed > peak_threshold
    )
    rows, columns = np.where(peaks)
    inside = (
        (rows >= radius)
        & (rows < average.shape[0] - radius)
        & (columns >= radius)
        & (columns < average.shape[1] - radius)
    )
    centres = np.column_stack([rows[inside], columns[inside]])
    if len(centres) == 0:
        raise ValueError("No sites detected. Check the image and peak threshold.")
    return centres


def resample_indices(n, rng, block_size=1):
    # The same trial indices are used for every site, retaining inter-site correlation.
    if block_size == 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
    return ((starts[:, None] + np.arange(block_size)) % n).ravel()[:n]


def bootstrap_sites(results, repeats=200, seed=2026, block_size=1):
    if repeats < 0 or block_size < 1:
        raise ValueError("Bootstrap count must be nonnegative and block size positive.")
    valid = [r for r in results if "signals" in r]
    if not valid:
        raise RuntimeError("No sites could be fitted.")
    n = len(valid[0]["signals"])
    if block_size > n:
        raise ValueError("Block size cannot exceed the number of trials.")
    rng = np.random.default_rng(seed)
    metrics = ("loading_fraction", "p_atom", "threshold", "fidelity")
    samples = {r["site"]: {m: [] for m in metrics} for r in valid}
    pooled = []
    for b in range(repeats):
        indices = resample_indices(n, rng, block_size)
        occupied_this_repeat = []
        for r in valid:
            try:
                fit = fit_signals(
                    r["signals"][indices],
                    initial_parameters=r["parameters"],
                    fixed_edges=r["edges"],
                )
                if fit["separation"] < 2 or not 0.02 < fit["p_atom"] < 0.98:
                    continue
                for metric in metrics:
                    samples[r["site"]][metric].append(fit[metric])
                occupied_this_repeat.append(fit["loading_fraction"])
            except (ValueError, RuntimeError, FloatingPointError):
                continue
        if len(occupied_this_repeat) == len(valid):
            pooled.append(np.mean(occupied_this_repeat))
        if (b + 1) % 20 == 0 or b + 1 == repeats:
            print(f"Bootstrap {b + 1}/{repeats}", flush=True)

    for r in valid:
        draws = samples[r["site"]]
        success = len(draws["threshold"])
        r["bootstrap_successes"] = success
        if repeats and success < 0.8 * repeats:
            r["flags"].append("unstable_bootstrap_fit")
        # Very small runs are for smoke testing, not reporting intervals.
        reliable = repeats >= 100 and success >= max(80, 0.8 * repeats)
        if repeats < 100:
            r["flags"].append("too_few_bootstraps_for_reporting")
        for metric in metrics:
            ci = (
                np.percentile(draws[metric], [2.5, 97.5])
                if reliable
                else [np.nan, np.nan]
            )
            r[metric + "_ci_low"], r[metric + "_ci_high"] = map(float, ci)
    pooled_ci = None
    if repeats >= 100 and len(pooled) >= max(80, 0.8 * repeats):
        pooled_ci = np.percentile(pooled, [2.5, 97.5]).tolist()
    return pooled_ci


def analyse_array(
    images, average, centres, radius=3, background=(110, 150, 35, 105), bins=30
):
    y0, y1, x0, x1 = background
    records = []
    for site, (row, column) in enumerate(centres, 1):
        row, column = int(row), int(column)
        bounds_y = (row - radius, row + radius + 1)
        bounds_x = (column - radius, column + radius + 1)
        if (
            bounds_y[0] < y1
            and bounds_y[1] > y0
            and bounds_x[0] < x1
            and bounds_x[1] > x0
        ):
            raise ValueError("Background region overlaps a selected site.")
        r = dict(
            site=site, row=row, column=column, rows=bounds_y, columns=bounds_x, flags=[]
        )
        try:
            signals = measure_signals(images, bounds_y, bounds_x, (y0, y1), (x0, x1))
            r.update(fit_signals(signals, bins=bins))
            # A rough warning about serial dependence, not a formal test.
            labels = r["occupied"].astype(float)
            lag1 = (
                np.corrcoef(labels[:-1], labels[1:])[0, 1] if np.std(labels) else np.nan
            )
            r["occupancy_lag1"] = float(lag1)
            if np.isfinite(lag1) and abs(lag1) > 2 / np.sqrt(len(labels)):
                r["flags"].append("possible_trial_dependence")
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            r["flags"] = ["fit_failed"]
            r["failure_reason"] = str(exc)
        records.append(r)
    return records


def sensitivity_checks(images, record, background, bins=30):
    # Check histogram choices, ROI sizes and a second atom-free background region.
    row, column = record["row"], record["column"]
    settings = [("bins", b, 3, background, b) for b in (20, 30, 40, 60)]
    settings += [
        ("roi_radius", radius, radius, background, bins) for radius in (2, 3, 4)
    ]
    settings += [("background", "lower_empty_region", 3, (155, 195, 35, 105), bins)]
    rows = []
    for kind, value, radius, bg, n_bins in settings:
        try:
            signal = measure_signals(
                images,
                (row - radius, row + radius + 1),
                (column - radius, column + radius + 1),
                bg[:2],
                bg[2:],
            )
            fit = fit_signals(signal, bins=n_bins)
            rows.append(
                dict(
                    check=kind,
                    value=value,
                    threshold=fit["threshold"],
                    loading_fraction=fit["loading_fraction"],
                    model_fidelity=fit["fidelity"],
                    flags=";".join(fit["flags"]),
                )
            )
        except (ValueError, RuntimeError) as exc:
            rows.append(dict(check=kind, value=value, flags=str(exc)))
    return rows


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_results(
    output, images, average, centres, results, filenames, config, pooled_ci, sensitivity
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "site",
        "row",
        "column",
        "loading_fraction",
        "loading_fraction_ci_low",
        "loading_fraction_ci_high",
        "p_atom",
        "p_atom_ci_low",
        "p_atom_ci_high",
        "threshold",
        "threshold_ci_low",
        "threshold_ci_high",
        "fidelity",
        "fidelity_ci_low",
        "fidelity_ci_high",
        "false_positive",
        "false_negative",
        "separation",
        "pearson_diagnostic",
        "occupancy_lag1",
        "bootstrap_successes",
        "flags",
        "failure_reason",
    ]
    rows = [{**r, "flags": ";".join(r["flags"])} for r in results]
    _write_csv(output / "site_results.csv", rows, fields)
    names = [f'site_{r["site"]:02d}' for r in results]
    with open(output / "signals.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image"] + names)
        for i, filename in enumerate(filenames):
            writer.writerow(
                [filename]
                + [r["signals"][i] if "signals" in r else "" for r in results]
            )
    with open(output / "occupancy.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image"] + names)
        for i, filename in enumerate(filenames):
            writer.writerow(
                [filename]
                + [int(r["occupied"][i]) if "occupied" in r else "" for r in results]
            )
    _write_csv(
        output / "sensitivity.csv",
        sensitivity,
        ["check", "value", "threshold", "loading_fraction", "model_fidelity", "flags"],
    )
    good = [r for r in results if "signals" in r]
    config.update(
        n_images=images.shape[2],
        n_sites=len(centres),
        successful_fits=len(good),
        review_sites=[r["site"] for r in results if r["flags"]],
        pooled_classified_fraction=float(
            np.mean([r["loading_fraction"] for r in good])
        ),
        pooled_classified_ci95=pooled_ci,
    )
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(average, cmap="inferno")
    for r in results:
        ax.add_patch(
            plt.Rectangle(
                (r["columns"][0] - 0.5, r["rows"][0] - 0.5),
                r["columns"][1] - r["columns"][0],
                r["rows"][1] - r["rows"][0],
                fill=False,
                edgecolor="cyan",
                linewidth=0.7,
            )
        )
        ax.text(
            r["column"] + 3, r["row"] - 3, str(r["site"]), color="white", fontsize=6
        )
    y0, y1, x0, x1 = config["background"]
    ax.add_patch(
        plt.Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="lime",
            label="Background",
        )
    )
    ax.set(
        xlabel="Column",
        ylabel="Row",
        title=f"{len(centres)} detected sites - check positions",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "detected_sites.png", dpi=160)
    plt.close(fig)

    ncols = 7
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 2.2 * nrows), squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        ax.set_title(f'Site {r["site"]}' + (" *" if r["flags"] else ""), fontsize=8)
        if "signals" in r:
            ax.hist(r["signals"], bins=r["edges"], alpha=0.6)
            x = np.linspace(r["edges"][0], r["edges"][-1], 300)
            ax.plot(x, double_gaussian(x, *r["parameters"]), color="black", linewidth=1)
            ax.axvline(r["threshold"], color="red", linestyle="--", linewidth=0.8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(results) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Per-site signal histograms (* = review flag); horizontal axis: corrected counts"
    )
    fig.tight_layout()
    fig.savefig(output / "all_histograms.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(
        axes,
        ("loading_fraction", "fidelity"),
        ("Classified loading fraction", "Gaussian-model fidelity"),
    ):
        dots = ax.scatter(
            [r["column"] for r in good],
            [r["row"] for r in good],
            c=[r[key] for r in good],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=140,
        )
        for r in good:
            if r["flags"]:
                ax.plot(r["column"], r["row"], "rx", markersize=12)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set(xlabel="Column", ylabel="Row", title=title)
        fig.colorbar(dots, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "array_maps.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for r in good:
        low, high = r["loading_fraction_ci_low"], r["loading_fraction_ci_high"]
        if np.isfinite(low):
            ax.vlines(r["site"], low, high, color="steelblue")
        ax.plot(r["site"], r["loading_fraction"], "rx" if r["flags"] else "k.")
    ax.set(
        xlabel="Site",
        ylabel="Classified loading fraction",
        ylim=(0, 1),
        title="Loading estimates and 95% bootstrap intervals (red = review)",
    )
    fig.tight_layout()
    fig.savefig(output / "loading_uncertainty.png", dpi=160)
    plt.close(fig)
    return config
