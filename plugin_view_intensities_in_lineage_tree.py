from functools import partial
from typing import Dict, Any, Optional

import matplotlib
from matplotlib.colors import Colormap

from organoid_tracker.core import UserError, Color
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.visualizer.lineage_tree_visualizer import LineageTreeVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//View-Intensity-colored lineage tree...": lambda: _show_lineage_tree(window)
    }


def _show_lineage_tree(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")
    intensity_keys = intensity_calculator.get_intensity_keys(experiment)
    if len(intensity_keys) == 0:
        raise UserError("No intensities recorded", "No intensities were recorded. Cannot plot anything.")

    dialog.popup_visualizer(window, IntensityLineageTreeVisualizer)


class IntensityLineageTreeVisualizer(LineageTreeVisualizer):
    """Shows lineage trees colored by the intensity of the positions."""

    _intensity_key: str
    _intensity_min_value: float = 0.0
    _intensity_max_value: float = 3.0
    _intensity_colormap: Colormap = matplotlib.cm.coolwarm
    _intensity_nan_color: Color = Color.black()

    def __init__(self, window: Window):
        super().__init__(window)

        # Set default intensity key
        intensity_keys = intensity_calculator.get_intensity_keys(self._experiment)
        if len(intensity_keys) > 0:
            self._intensity_key = intensity_keys[0]
        else:
            self._intensity_key = intensity_calculator.DEFAULT_INTENSITY_KEY

        # Default settings
        self._uncolor_lineages()
        self._display_deaths = False
        self._display_custom_colors = True

    def _get_custom_color_label(self) -> Optional[str]:
        return "intensities"

    def _get_lineage_line_width(self) -> float:
        return 3

    def get_extra_menu_options(self) -> Dict[str, Any]:
        intensity_keys = intensity_calculator.get_intensity_keys(self._experiment)

        options = {
            **super().get_extra_menu_options(),
            "Intensity//Colormap//Split-Blue to red": partial(self._set_colormap, "coolwarm"),
            "Intensity//Colormap//Split-Pink to green": partial(self._set_colormap, "PiYG"),
            "Intensity//Colormap//Split-Purple to green": partial(self._set_colormap, "PRGn"),
            "Intensity//Colormap//Single-Blue": partial(self._set_colormap, "Blues"),
            "Intensity//Colormap//Single-Gray": partial(self._set_colormap, "Grays"),
            "Intensity//Colormap//Single-Green": partial(self._set_colormap, "Greens"),
            "Intensity//Colormap//Single-Orange": partial(self._set_colormap, "Oranges"),
            "Intensity//Colormap//Single-Purple": partial(self._set_colormap, "Purples"),
            "Intensity//Colormap//Single-Red": partial(self._set_colormap, "Reds"),
            "Intensity//Colormap//Uniform-Cividis": partial(self._set_colormap, "cividis"),
            "Intensity//Colormap//Uniform-Inferno": partial(self._set_colormap, "inferno"),
            "Intensity//Colormap//Uniform-Magma": partial(self._set_colormap, "magma"),
            "Intensity//Colormap//Uniform-Plasma": partial(self._set_colormap, "plasma"),
            "Intensity//Colormap//Uniform-Viridis": partial(self._set_colormap, "viridis"),
        }
        if len(intensity_keys) > 1:
            for intensity_key in intensity_keys:
                options["Intensity//Intensity selector//" + intensity_key] = partial(self._switch_intensity_key, intensity_key)

        return options

    def _set_colormap(self, name: str):
        self._intensity_colormap = matplotlib.colormaps.get(name)
        self.update_status(f"Now coloring by the \"{name}\" colormap of Matplotlib.")
        self.draw_view()

    def _switch_intensity_key(self, intensity_key: str):
        self._uncolor_lineages()
        self._intensity_key = intensity_key
        self._display_custom_colors = True
        self.update_status(f"Now coloring by the intensity values stored under \"{intensity_key}\"; "
                           f"turned off other lineage coloring")
        self.draw_view()

    def _get_custom_color(self, position: Position) -> Optional[Color]:
        intensity = intensity_calculator.get_normalized_intensity(self._experiment, position,
                                                                  intensity_key=self._intensity_key)
        if intensity is None:
            return self._intensity_nan_color

        intensity = (intensity - self._intensity_min_value) / (self._intensity_max_value - self._intensity_min_value)
        r, g, b, a = self._intensity_colormap(intensity)
        return Color(int(r * 255), int(g * 255), int(b * 255))
