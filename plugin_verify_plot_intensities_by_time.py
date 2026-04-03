from collections import defaultdict
from typing import Any

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core.resolution import ImageTimings
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Intensity//Record-Verify intensities//LineGraph-Plot intensities by time...": lambda: _plot_intensities_by_t(window)
    }


def _time_points_to_hours(time_point_numbers: ndarray, timings: ImageTimings) -> ndarray:
    t_values_h = numpy.empty_like(time_point_numbers, dtype=numpy.float64)
    for i in range(len(time_point_numbers)):
        t_values_h[i] = timings.get_time_h_since_start(time_point_numbers[i])
    return t_values_h


def _draw_intensities_by_t(figure: Figure, timings: ImageTimings | None, intensities_by_name_and_t: dict[str, dict[int, list[float]]]):
    ax: Axes = figure.gca()

    i = 0
    for intensity_key, values_by_t in intensities_by_name_and_t.items():
        time_point_numbers = numpy.arange(min(values_by_t.keys()), max(values_by_t.keys()) + 1)
        intensity_means = numpy.full_like(time_point_numbers, fill_value=numpy.nan, dtype=numpy.float64)
        intensity_stds = numpy.full_like(time_point_numbers, fill_value=numpy.nan, dtype=numpy.float64)
        for t, values in values_by_t.items():
            time_index = t - time_point_numbers[0]
            intensity_means[time_index] = numpy.mean(values)
            intensity_stds[time_index] = numpy.std(values, ddof=1)

        # Filter out NaNs (we don't have positions for that time point)
        nan_values = numpy.isnan(intensity_means)
        time_point_numbers = time_point_numbers[~nan_values]
        intensity_means = intensity_means[~nan_values]
        intensity_stds = intensity_stds[~nan_values]

        if timings is not None:
            time_point_numbers = _time_points_to_hours(time_point_numbers, timings)

        color = SANDER_APPROVED_COLORS[i % len(SANDER_APPROVED_COLORS)]
        ax.plot(time_point_numbers, intensity_means, label=intensity_key, color=color, linewidth=3)
        ax.fill_between(time_point_numbers, intensity_means - intensity_stds, intensity_means + intensity_stds, color=color, alpha=0.4)
        i += 1

    ax.set_ylabel("Intensity/px (a.u.)")
    if timings is None:
        ax.set_xlabel("Time (time points)")
    else:
        ax.set_xlabel("Time (h)")
    if len(intensities_by_name_and_t) > 1:
        ax.legend()


def _plot_intensities_by_t(window: Window):
    intensities_by_name_and_t = dict()

    timings = None
    make_timings_unavailable = False

    for experiment in window.get_active_experiments():
        if experiment.images.has_timings():
            new_timings = experiment.images.timings()
            if timings is not None and timings != new_timings:
                make_timings_unavailable = True  # Found an experiment with different timings than another one
            timings = new_timings
        else:
            make_timings_unavailable = True  # Found an experiment without timings

        for intensity_key in intensity_calculator.get_regular_intensity_keys(experiment):
            if intensity_key not in intensities_by_name_and_t:
                intensities_by_name_and_t[intensity_key] = defaultdict(list)

            for position, _ in experiment.positions.find_all_positions_with_data(intensity_key):
                intensity = intensity_calculator.get_normalized_intensity(experiment, position,
                                                                          intensity_key=intensity_key, per_pixel=True)
                if intensity is None:
                    continue
                intensities_by_name_and_t[intensity_key][position.time_point_number()].append(intensity)

    if make_timings_unavailable:
        timings = None

    dialog.popup_figure(window, lambda figure: _draw_intensities_by_t(figure, timings, intensities_by_name_and_t))


