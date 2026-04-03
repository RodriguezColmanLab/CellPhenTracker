from functools import partial
from typing import Any

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Intensity//Record-Filter intensities//By intensity...": lambda: _view_intensity_filtering(window)
    }


def _view_intensity_filtering(window: Window):
    activate(_IntensityFilteringVisualizer(window))


class _IntensityFilteringVisualizer(ExitableImageVisualizer):
    """You can use this screen to filter out intensities outside of a certain range. This is useful for example to
    filter out darkest cells (useful in calculating ratiometric intensities). We only filter out on the track level,
    so either all intensities of a track are filtered out, or none. You can set at which percentage of outlier
    intensities, the entire track is filtered out.

    Use the Parameters menu to set the intensity range, and then use the Edit menu to apply the filtering.
    """

    _intensity_key: str = "intensity"
    _min_intensity: float = 0
    _max_intensity: float = 1
    _max_percentage_per_track: float = 50

    def __init__(self, window: Window):
        super().__init__(window)

        self._auto_adjust_min_max()

    def _auto_adjust_min_max(self):
        """Sets a default min and max intensity based on the intensities in this time point,
        so that the user has a good starting point for filtering."""
        intensity_key = self._get_intensity_key()
        intensities = []
        for position in self._experiment.positions.of_time_point(self._time_point):
            intensity = intensity_calculator.get_normalized_intensity(self._experiment, position, intensity_key=intensity_key)
            if intensity is not None:
                intensities.append(intensity)
        if len(intensities) > 0:
            # Set a wider range, so that intensities from other time points that are higher or lower, are not
            # immediately filtered out
            self._min_intensity = min(intensities) * 0.66
            self._max_intensity = max(intensities) * 1.5

    def _get_available_intensity_keys(self) -> set[str]:
        intensity_keys = set()
        for experiment in self._window.get_active_experiments():
            intensity_keys.update(intensity_calculator.get_intensity_keys(experiment))
        return intensity_keys

    def _remove_intensities_outside_range(self):
        if not dialog.popup_message_cancellable("Intensity filtering",
                                                f"Are you sure you want to filter out intensities outside the range? You cannot undo this action."):
            return

        intensity_key = self._get_intensity_key()

        track_removed_count = 0
        for tab in self._window.get_gui_experiment().get_active_tabs():
            experiment = tab.experiment
            for track in tab.experiment.links.find_all_tracks():
                out_of_range_count = 0
                total_count = 0
                for position in track.positions():
                    intensity = intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=intensity_key)
                    if intensity is not None:
                        total_count += 1
                        if intensity < self._min_intensity or intensity > self._max_intensity:
                            out_of_range_count += 1

                if total_count > 0 and (out_of_range_count / total_count) * 100 >= self._max_percentage_per_track:
                    # Too many intensities are out of range, filter out the entire track
                    self._remove_intensities_of_track(experiment, track, intensity_key)
                    track_removed_count += 1
            tab.undo_redo.clear()

        self.update_status(f"Removed intensities of {track_removed_count} tracks that had more than "
                           f"{self._max_percentage_per_track:.0f}% of their intensities outside the range "
                           f"[{self._min_intensity:.2f}, {self._max_intensity:.2f}].")

    def _remove_intensities_of_track(self, experiment: Experiment, track: LinkingTrack, intensity_key: str):
        for position in track.positions():
            experiment.positions.set_position_data(position, intensity_key, None)
            experiment.positions.set_position_data(position, intensity_key + "_volume", None)

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if dt != 0 or abs(dz) > 2:
            return True  # Only look at the current time point and z-slice

        intensity_key = self._get_intensity_key()
        if intensity_key is None:
            return True  # No intensity keys available, so we don't filter anything out

        intensity = intensity_calculator.get_normalized_intensity(self._experiment, position, intensity_key=intensity_key)
        if intensity is None:
            return True  # No intensity for this position, so we don't filter it out

        text_color = "darkred"
        background_color = (1, 1, 1, 0.8)
        if self._min_intensity <= intensity <= self._max_intensity:
            text_color = "lime"
            background_color = (0.3, 0.3, 0.3, 0.8)

        if abs(intensity) < 10:
            intensity_text = f"{intensity:.2f}"
        elif abs(intensity) < 100:
            intensity_text = f"{intensity:.1f}"
        elif abs(intensity) < 10000:
            intensity_text = f"{intensity:.0f}"
        elif abs(intensity) < 1_000_000:
            intensity_text = f"{intensity / 1000:.0f}k"
        else:
            intensity_text = f"{intensity:.1e}"
        self._draw_annotation(position, intensity_text, text_color=text_color, background_color=background_color)
        return False

    def _get_figure_title(self) -> str:
        return f"Time point {self._time_point.time_point_number()}    (z={self._get_figure_title_z_str()}, " \
               f"i={self._get_intensity_key()})"

    def get_extra_menu_options(self) -> dict[str, Any]:
        menu_options = {
            **super().get_extra_menu_options(),
            "Edit//Apply-Apply intensity filtering": self._remove_intensities_outside_range,
            "Parameters//Intensity-Set minimum intensity...": self._set_min_intensity,
            "Parameters//Intensity-Set maximum intensity...": self._set_max_intensity,
            "Parameters//Intensity-Set track filtering percentage...": self._set_max_percentage_per_track,
        }

        intensity_keys = self._get_available_intensity_keys()
        if len(intensity_keys) > 1:
            # Add a menu to select the intensity key
            for intensity_key in intensity_keys:
                menu_options["Parameters//Selector-Select intensity//" + intensity_key] = partial(self._set_intensity_key, intensity_key)
        return menu_options

    def _get_intensity_key(self) -> str:
        intensity_keys = self._get_available_intensity_keys()
        if len(intensity_keys) == 1:
            # Ignore selection if we only have one option
            return next(iter(intensity_keys))
        if self._intensity_key not in intensity_keys and len(intensity_keys) > 0:
            # Key not available, select one from the available keys
            return next(iter(intensity_keys))
        return self._intensity_key  # Chose the one the user picked

    def _set_min_intensity(self):
        min_intensity = dialog.prompt_float("Set minimum intensity", "Minimum intensity:", default=self._min_intensity, decimals=2)
        if min_intensity is None:
            return  # User cancelled
        self._min_intensity = min_intensity
        self.update_status(f"Minimum intensity set to {self._min_intensity:.2f}. If you're happy with the filtering, use the Edit menu to apply the filtering to the experiment.")
        self.draw_view()

    def _set_max_intensity(self):
        max_intensity = dialog.prompt_float("Set maximum intensity", "Maximum intensity:", default=self._max_intensity, decimals=2)
        if max_intensity is None:
            return  # User cancelled
        self._max_intensity = max_intensity
        self.update_status(f"Maximum intensity set to {self._max_intensity:.2f}. If you're happy with the filtering, use the Edit menu to apply the filtering to the experiment.")
        self.draw_view()

    def _set_max_percentage_per_track(self):
        max_percentage_per_track = dialog.prompt_float("Set track filtering percentage",
             "At which percentage of time points that fall outside the range, should the entire track be filtered out?",
             default=self._max_percentage_per_track, decimals=0,
             minimum=0, maximum=100)
        if max_percentage_per_track is None:
            return  # User cancelled
        self._max_percentage_per_track = max_percentage_per_track
        self.update_status(f"Track filtering percentage set to {self._max_percentage_per_track:.0f}%.")
        self.draw_view()

    def _set_intensity_key(self, intensity_key: str):
        self._intensity_key = intensity_key
        self._auto_adjust_min_max()
        self.update_status(f"Intensity key set to {intensity_key}.")
        self.draw_view()