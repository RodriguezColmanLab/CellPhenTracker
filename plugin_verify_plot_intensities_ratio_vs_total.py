from collections import defaultdict
from functools import partial
from typing import Any, NamedTuple

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core.resolution import ImageTimings
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS

class _RatiosOfExperiment(NamedTuple):
    experiment_name: str
    totals: list[float]
    ratios: list[float]


def _find_ratiometric_intensities(window: Window) -> list[str]:
    ratiometric_intensities = set()
    for experiment in window.get_active_experiments():
        ratiometric_intensities.update(set(intensity_calculator.get_ratiometric_intensity_keys(experiment)))
    return sorted(ratiometric_intensities)


def get_menu_items(window: Window) -> dict[str, Any]:
    ratiometric_intensities = _find_ratiometric_intensities(window)

    return_dict = dict()
    for ratiometric_intensity_key in ratiometric_intensities:
        return_dict["Intensity//Record-Verify intensities//Ratiometric-Plot ratio versus total//" + ratiometric_intensity_key]\
            = partial(_plot_intensities_ratio_vs_total, window, ratiometric_intensity_key)

    return return_dict


def _time_points_to_hours(time_point_numbers: ndarray, timings: ImageTimings) -> ndarray:
    t_values_h = numpy.empty_like(time_point_numbers, dtype=numpy.float64)
    for i in range(len(time_point_numbers)):
        t_values_h[i] = timings.get_time_h_since_start(time_point_numbers[i])
    return t_values_h


def _draw_intensities(figure: Figure, intensities_of_all_experiments: list[_RatiosOfExperiment]):
    ax: Axes = figure.gca()

    i = 0
    for intensities_of_experiment in intensities_of_all_experiments:
        ax.scatter(intensities_of_experiment.ratios, intensities_of_experiment.totals,
                   s=20, marker="s", lw=0, alpha=0.8, color=SANDER_APPROVED_COLORS[i % len(SANDER_APPROVED_COLORS)],
                   label=intensities_of_experiment.experiment_name)
        i += 1

    if len(intensities_of_all_experiments) > 1:
        ax.legend()
    ax.set_xlabel("Ratiometric intensity (a.u.)")
    ax.set_ylabel("Summed intensity (a.u.)")
    ax.set_title("Ratiometric vs. total intensity\n(all time points)")

def _plot_intensities_ratio_vs_total(window: Window, intensity_key: str):
    all_ratios = list()

    for experiment in window.get_active_experiments():
        individual_keys = intensity_calculator.get_intensities_for_ratiometric_intensity(experiment, intensity_key)
        if individual_keys is None:
            continue  # No data for this experiment

        sums = list()
        ratios = list()
        for position, _ in experiment.positions.find_all_positions_with_data(individual_keys[0]):
            intensity_1 = intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=individual_keys[0])
            intensity_2 = intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=individual_keys[1])
            intensity_ratio = intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=intensity_key)

            if intensity_1 is None or intensity_2 is None or intensity_ratio is None:
                continue

            sums.append(intensity_1 + intensity_2)
            ratios.append(intensity_ratio)

        if len(sums) > 0:
            all_ratios.append(_RatiosOfExperiment(totals=sums, ratios=ratios, experiment_name=str(experiment.name)))

    dialog.popup_figure(window, lambda figure: _draw_intensities(figure, all_ratios))


