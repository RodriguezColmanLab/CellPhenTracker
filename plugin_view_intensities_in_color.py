from typing import Optional, Dict, Any, Set, List

import matplotlib.cm
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from organoid_tracker.core import UserError, Color, max_none
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.moving_average import MovingAverage
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer

_AVERAGING_WINDOW_H = 4
_STEP_SIZE_H = 0.2


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//View-View intensities in color...": lambda: _view_intensities(window)
    }


def _view_intensities(window: Window):
    activate(_IntensityInColorPlotter(window))



class _IntensityInColorPlotter(ExitableImageVisualizer):
    """Shows cells colored by their intensity, relative to other cells in this time point. Click on a cell to view
    its intensity as a number."""

    _maximum_intensity: Optional[float]
    _intensity_colormap: Colormap = matplotlib.cm.get_cmap("jet")
    _intensity_key: str

    def __init__(self, window: Window):
        super().__init__(window)
        self._intensity_key = self._check_for_intensities()
        self._experiment.links.sort_tracks_by_x()

    def get_extra_menu_options(self) -> Dict[str, Any]:
        intensity_keys = self._get_available_intensity_keys()
        if len(intensity_keys) == 1 and next(iter(intensity_keys)) == self._intensity_key:
            return dict()  # No need to show a selection menu

        return_value = dict()
        for intensity_key in intensity_keys:
            return_value["Intensity//Selector-" + intensity_key] = lambda: self._set_intensity_key(intensity_key)
        return return_value

    def _set_intensity_key(self, new_key: str):
        self._intensity_key = new_key
        self._calculate_time_point_metadata()
        self.draw_view()
        self.update_status("Now viewing intensities stored with the key \"" + self._intensity_key + "\" .")

    def _must_show_other_time_points(self) -> bool:
        return False

    def _calculate_time_point_metadata(self):
        # Calculates the maximum intensity of this time point
        max_intensity = None
        for position in self._experiment.positions.of_time_point(self._time_point):
            max_intensity = max_none(max_intensity, intensity_calculator.get_normalized_intensity(self._experiment, position, intensity_key=self._intensity_key))
        self._maximum_intensity = max_intensity

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        # Draws a circle in a color based on the intensity
        if self._maximum_intensity is None:
            return True

        if dt != 0:
            return False  # ignore other time points
        intensity = intensity_calculator.get_normalized_intensity(self._experiment, position)
        if intensity is None:
            return False
        intensity_fraction = float(intensity / self._maximum_intensity)
        color = self._intensity_colormap(intensity_fraction)
        self._ax.plot(position.x, position.y, 'o', markersize=15 - abs(dz), color=color, markeredgecolor=color, markeredgewidth=5)
        return False

    def _on_mouse_click(self, event: MouseEvent):
        # Prints the intensity of a cell
        self._check_for_intensities()

        selected_position = self._get_position_at(event.xdata, event.ydata)
        if selected_position is None or self._maximum_intensity is None:
            return

        intensity = intensity_calculator.get_normalized_intensity(self._experiment, selected_position)
        percentage = intensity / self._maximum_intensity * 100
        self.update_status(f"The intensity of {selected_position} was measured as {intensity:.2f}, which is {percentage:.1f}% of"
                           f"the maximum of this time point")

    def _get_available_intensity_keys(self) -> Set[str]:
        intensity_keys = set()
        for experiment in self._window.get_active_experiments():
            intensity_keys |= set(intensity_calculator.get_intensity_keys(experiment))
        return intensity_keys

    def _check_for_intensities(self) -> str:
        """Displays a message if there are no recorded intensities. Returns the key for one intensity."""
        intensity_keys = self._get_available_intensity_keys()
        if len(intensity_keys) == 0:
            dialog.popup_error("No intensities recorded", "No intensities are recorded. Please do so"
                                                          " first from the main screen.")
        return intensity_keys.pop()