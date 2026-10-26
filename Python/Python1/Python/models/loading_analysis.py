"""Pixel-sum analysis for one tweezer across repeated images.

Images must have shape (rows, columns, number_of_images), as returned by
load_tga_folder(). The supplied example coordinates are for the S41 data
after the original loader's ten-row crop and the 16-bit reading correction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm


def _check_region(shape, rows, columns):
    """Reject coordinates outside the image or an empty selection."""
    if not (
        0 <= rows[0] < rows[1] <= shape[0] and 0 <= columns[0] < columns[1] <= shape[1]
    ):
        raise ValueError("Region must be inside the image, with increasing bounds.")


def measure_signals(images, rows, columns, background_rows, background_columns):
    """Return one background-corrected pixel sum for each image."""
    images = np.asarray(images, dtype=float)
    if images.ndim != 3 or images.shape[2] == 0:
        raise ValueError("Expected images with shape (rows, columns, trials).")
    _check_region(images.shape, rows, columns)
    _check_region(images.shape, background_rows, background_columns)

    roi = images[rows[0] : rows[1], columns[0] : columns[1], :]
    background_region = images[
        background_rows[0] : background_rows[1],
        background_columns[0] : background_columns[1],
        :,
    ]

    # Estimate background per pixel separately in each image.
    background = np.mean(background_region, axis=(0, 1))
    pixel_count = roi.shape[0] * roi.shape[1]
    signals = np.sum(roi, axis=(0, 1)) - pixel_count * background
    return signals


def double_gaussian(x, a0, m0, s0, a1, m1, s1):
    """Two 1D histogram peaks, rather than a 2D fit to an image spot."""
    empty = a0 * np.exp(-0.5 * ((x - m0) / s0) ** 2)
    occupied = a1 * np.exp(-0.5 * ((x - m1) / s1) ** 2)
    return empty + occupied


def fit_signals(signals, bins=30, initial_parameters=None, fixed_edges=None):
    """Fit the histogram and estimate a threshold and classification errors.

    These estimates assume two Gaussian populations. A successful numerical
    fit alone does not establish that this model describes the measurements.
    """
    signals = np.asarray(signals, dtype=float)
    if (
        signals.ndim != 1
        or len(signals) < 20
        or not np.isfinite(signals).all()
        or np.ptp(signals) <= 1
    ):
        raise ValueError("Provide at least 20 finite, nonconstant signal values.")

    counts, edges = np.histogram(
        signals, bins=bins if fixed_edges is None else fixed_edges
    )
    centres = (edges[:-1] + edges[1:]) / 2

    # Initial guesses: lower and upper quarters of the measured signals.
    width = max(np.std(signals) / 4, 1.01)
    guess = [
        max(counts),
        np.percentile(signals, 25),
        width,
        max(counts),
        np.percentile(signals, 75),
        width,
    ]
    lower = [0, signals.min(), 1, 0, signals.min(), 1]
    upper = [
        np.inf,
        signals.max(),
        np.ptp(signals),
        np.inf,
        signals.max(),
        np.ptp(signals),
    ]
    # Several starting guesses reduce the chance of fitting the same peak twice.
    if initial_parameters is None:
        guesses = [guess]
        for fraction in (0.15, 0.5):
            guesses.append(
                [
                    max(counts),
                    np.percentile(signals, 15),
                    max(np.std(signals) * fraction, 1.01),
                    max(counts),
                    np.percentile(signals, 85),
                    max(np.std(signals) * fraction, 1.01),
                ]
            )
    else:
        guesses = [
            np.clip(initial_parameters, np.array(lower) + 1e-7, np.array(upper) - 1e-7)
        ]
    fits = []
    import warnings
    from scipy.optimize import OptimizeWarning

    for start in guesses:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                p, _ = curve_fit(
                    double_gaussian,
                    centres,
                    counts,
                    p0=start,
                    bounds=(lower, upper),
                    maxfev=10000,
                    x_scale="jac",
                )
            residual = np.sum((counts - double_gaussian(centres, *p)) ** 2)
            fits.append((residual, p))
        except (RuntimeError, ValueError, FloatingPointError):
            continue
    if not fits:
        raise RuntimeError("Two-population fit did not converge.")
    parameters = min(fits, key=lambda item: item[0])[1]

    # The population with the lower mean is labelled empty.
    groups = sorted([parameters[:3], parameters[3:]], key=lambda g: g[1])
    a0, m0, s0 = groups[0]
    a1, m1, s1 = groups[1]

    # Gaussian areas are proportional to height times width.
    p_atom = (a1 * s1) / (a0 * s0 + a1 * s1)
    thresholds = np.linspace(m0, m1, 2000)
    false_positive = norm.sf(thresholds, loc=m0, scale=s0)
    false_negative = norm.cdf(thresholds, loc=m1, scale=s1)

    # Choose the best single threshold between the two fitted means.
    error = (1 - p_atom) * false_positive + p_atom * false_negative
    best = np.argmin(error)
    threshold = thresholds[best]
    occupied = signals > threshold

    separation = (m1 - m0) / np.sqrt((s0**2 + s1**2) / 2)
    # Diagnostic only: sparse bins and estimated parameters complicate formal tests.
    expected = len(signals) * (
        (1 - p_atom) * np.diff(norm.cdf(edges, m0, s0))
        + p_atom * np.diff(norm.cdf(edges, m1, s1))
    )
    populated = expected >= 5
    pearson = np.sum(
        (counts[populated] - expected[populated]) ** 2 / expected[populated]
    ) / max(np.sum(populated) - 6, 1)
    flags = []
    if separation < 2:
        flags.append("poor_peak_separation")
    if not 0.02 < p_atom < 0.98:
        flags.append("very_small_population")
    if pearson > 3:
        flags.append("histogram_model_mismatch")

    return {
        "signals": signals,
        "counts": counts,
        "edges": edges,
        "parameters": np.concatenate(groups),
        "threshold": float(threshold),
        "occupied": occupied,
        "loading_fraction": float(np.mean(occupied)),
        "p_atom": float(p_atom),
        "fidelity": float(1 - error[best]),
        "false_positive": float(false_positive[best]),
        "false_negative": float(false_negative[best]),
        "separation": float(separation),
        "pearson_diagnostic": float(pearson),
        "flags": flags,
    }


def analyse_tweezer(
    images, rows, columns, background_rows, background_columns, bins=30
):
    """Measure one site, fit its histogram and return the results."""
    signals = measure_signals(
        images, rows, columns, background_rows, background_columns
    )
    result = fit_signals(signals, bins=bins)
    result.update(
        rows=rows,
        columns=columns,
        background_rows=background_rows,
        background_columns=background_columns,
    )
    return result


def plot_analysis(average, result):
    """Create plots; the caller uses plt.show() or saves the figures."""
    rows, columns = result["rows"], result["columns"]
    fig_image, ax = plt.subplots()
    image = ax.imshow(average, cmap="inferno")
    ax.add_patch(
        plt.Rectangle(
            (columns[0] - 0.5, rows[0] - 0.5),
            columns[1] - columns[0],
            rows[1] - rows[0],
            fill=False,
            edgecolor="cyan",
            linewidth=2,
        )
    )
    ax.set(xlabel="Column", ylabel="Row", title="Selected tweezer")
    fig_image.colorbar(image, ax=ax, label="Mean camera counts")
    fig_image.tight_layout()

    fig_histogram, ax = plt.subplots()
    ax.hist(result["signals"], bins=result["edges"], alpha=0.6, label="Measurements")
    x = np.linspace(result["edges"][0], result["edges"][-1], 1000)
    a0, m0, s0, a1, m1, s1 = result["parameters"]
    ax.plot(
        x,
        double_gaussian(x, *result["parameters"]),
        color="black",
        label="Two-Gaussian fit",
    )
    ax.plot(
        x,
        a0 * np.exp(-0.5 * ((x - m0) / s0) ** 2),
        linestyle=":",
        label="Empty population",
    )
    ax.plot(
        x,
        a1 * np.exp(-0.5 * ((x - m1) / s1) ** 2),
        linestyle=":",
        label="Occupied population",
    )
    ax.axvline(result["threshold"], color="red", linestyle="--", label="Threshold")
    ax.set(
        xlabel="Background-corrected summed counts",
        ylabel="Number of images",
        title="One tweezer: signal histogram",
    )
    ax.legend()
    fig_histogram.tight_layout()
    return fig_image, fig_histogram
