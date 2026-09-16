import math
from collections import defaultdict
from typing import Any

import numpy
import scipy.stats
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Intensity//Record-Verify intensities//LineGraph-Plot intensities by z...": lambda: _plot_intensities_by_z(window)
    }


def _draw_intensities_by_z(figure: Figure, intensities_by_name_and_z: dict[str, dict[int, list[float]]]):
    ax: Axes = figure.gca()

    i = 0
    overall_min_z = None
    overall_max_z = None

    random = numpy.random.Generator(numpy.random.MT19937(seed=1))
    for intensity_key, values_by_z in intensities_by_name_and_z.items():
        color = SANDER_APPROVED_COLORS[i % len(SANDER_APPROVED_COLORS)]

        min_z = min(values_by_z.keys())
        max_z = max(values_by_z.keys())

        if overall_min_z is None or min_z < overall_min_z:
            overall_min_z = min_z
        if overall_max_z is None or max_z > overall_max_z:
            overall_max_z = max_z

        r = _plot_logarithmic_fit(ax, values_by_z, color=color)
        label = f"{intensity_key} (r={r:.2f})" if not math.isnan(r) else intensity_key
        for z in range(min_z, max_z + 1):
            if z not in values_by_z:
                continue

            values = values_by_z[z]
            ax.scatter(random.normal(loc=z, scale=0.1, size=len(values)), values, label=label if z == min_z else None, color=color, alpha=1, lw=0, s=10)
        i += 1

    if overall_min_z is not None and overall_max_z is not None:
        ax.set_xticks(range(overall_min_z, overall_max_z + 1), minor=True)
        width = overall_max_z - overall_min_z
        ax.set_xlim(overall_min_z - width / 10, overall_max_z + width / 10)
    ax.set_ylabel("Intensity/px (a.u.)")
    ax.set_xlabel("Z (px)")
    if len(intensities_by_name_and_z) > 1:
        ax.legend()
    figure.tight_layout()

def _plot_logarithmic_fit(ax: Axes, values_by_z: dict[int, list[float]], color: str):
    if len(values_by_z) < 4:
        return float("NaN") # Not enough data points to fit a logarithmic curve

    z_values = []
    intensities_log = []
    for z, values in values_by_z.items():
        if min(values) <= 0:
            return float("NaN") # Cannot take logarithm of non-positive values
        z_values.extend([z] * len(values))
        intensities_log.extend([math.log(v) for v in values])

    linear_fit = scipy.stats.linregress(z_values, intensities_log)

    plotting_z_values = numpy.arange(min(z_values), max(z_values) + 1)
    plotting_intensities = numpy.exp(linear_fit.slope * plotting_z_values + linear_fit.intercept)
    ax.plot(plotting_z_values, plotting_intensities, color=color, linestyle='--', linewidth=2)

    return linear_fit.rvalue

def _plot_intensities_by_z(window: Window):
    intensities_by_name_and_z = dict()
    for experiment in window.get_active_experiments():
        for intensity_key in intensity_calculator.get_regular_intensity_keys(experiment):
            if intensity_key not in intensities_by_name_and_z:
                intensities_by_name_and_z[intensity_key] = defaultdict(list)

            for position, _ in experiment.positions.find_all_positions_with_data(intensity_key):
                intensity = intensity_calculator.get_normalized_intensity(experiment, position,
                                                                          intensity_key=intensity_key, per_pixel=True)
                if intensity is None:
                    continue
                z = int(round(position.z))
                intensities_by_name_and_z[intensity_key][z].append(intensity)

    dialog.popup_figure(window, lambda figure: _draw_intensities_by_z(figure, intensities_by_name_and_z), size_cm=(20, 10))


