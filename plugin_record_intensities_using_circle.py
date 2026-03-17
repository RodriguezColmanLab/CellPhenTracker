"""Uses a simple circle of a given radius for segmentation"""
import math
from typing import Optional, Dict, Any, Set

from matplotlib.patches import Ellipse

from organoid_tracker.core import UserError, bounding_box
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.mask import Mask
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog, worker_job
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Record intensities//Record intensity using circle...": lambda: _view_intensities(window)
    }


def _view_intensities(window: Window):
    activate(_CircleSegmentationVisualizer(window))


def _get_intensity(position: Position, intensity_image: Image, mask: Mask) -> Optional[int]:
    mask.center_around(position)
    if mask.count_pixels() == 0:
        return None
    masked_image = mask.create_masked_image(intensity_image)
    return int(masked_image.sum())


def _create_circular_mask(radius_um: float, resolution: ImageResolution) -> Mask:
    """Creates a mask that is spherical in micrometers. If the resolution is not the same in the x, y and z directions,
    this sphere will appear as a spheroid in the images."""
    radius_x_px = math.ceil(radius_um / resolution.pixel_size_x_um)
    radius_y_px = math.ceil(radius_um / resolution.pixel_size_y_um)
    mask = Mask(bounding_box.ONE.expanded(radius_x_px, radius_y_px, 0))

    # Evaluate the spheroid function to draw it
    mask.add_from_function(lambda x, y, z:
                           x ** 2 / radius_x_px ** 2 + y ** 2 / radius_y_px ** 2 <= 1)

    return mask


class _RecordIntensitiesJob(WorkerJob):
    """Records the intensities of all positions."""

    _radius_um: float
    _measurement_channel: ImageChannel
    _intensity_key: str

    def __init__(self, radius_um: float, measurement_channel: ImageChannel, intensity_key: str):
        self._radius_um = radius_um
        self._measurement_channel = measurement_channel
        self._intensity_key = intensity_key

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(positions=True, images=True)

    def gather_data(self, experiment_copy: Experiment) -> Dict[Position, int]:
        results = dict()
        circular_mask = _create_circular_mask(self._radius_um, experiment_copy.images.resolution())
        for time_point in experiment_copy.positions.time_points():
            positions = list(experiment_copy.positions.of_time_point(time_point))
            if len(positions) == 0:
                continue  # Skip this time point

            # Load images
            measurement_image = experiment_copy.images.get_image(time_point, self._measurement_channel)
            if measurement_image is None:
                continue  # Skip this time point, image is missing

            # Calculate intensities
            for position in positions:
                intensity = _get_intensity(position, measurement_image, circular_mask)
                if intensity is not None:
                    results[position] = intensity
        return results

    def use_data(self, tab: SingleGuiTab, data: Dict[Position, int]):
        intensities = data

        # Record volumes too, for administrative purposes
        resolution = tab.experiment.images.resolution()
        circle_volume_px3 = _create_circular_mask(self._radius_um, resolution).count_pixels()

        volumes_px = dict(((position, circle_volume_px3) for position in intensities.keys()))
        intensity_calculator.set_raw_intensities(tab.experiment, intensities, volumes_px,
                                                 intensity_key=self._intensity_key)
        tab.undo_redo.mark_unsaved_changes()

    def on_finished(self, result: Any):
        dialog.popup_message("Intensities recorded", "All intensities have been recorded.\n\n"
                                                     "Your next step is likely to set a normalization. This can be\n"
                                                     "done from the Intensity menu in the main screen of the program.")


class _CircleSegmentationVisualizer(ExitableImageVisualizer):
    """First, specify the membrane and measurement channels in the Parameters menu.
    Then, record the intensities of each cell. If you are happy with the masks, then
    use Edit -> Record intensities."""

    _measurement_channel: Optional[ImageChannel] = None
    _nucleus_radius_um: float = 3
    _intensity_key: str

    def __init__(self, window: Window):
        super().__init__(window)
        self._intensity_key = intensity_calculator.DEFAULT_INTENSITY_KEY
        self._display_settings.max_intensity_projection = False

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Channels-Record intensities...": self._record_intensities,
            "Parameters//Channel-Set measurement channel...": self._set_channel,
            "Parameters//Other-Set nucleus radius...": self._set_nucleus_radius,
            "Parameters//Other-Set storage key...": self._set_intensity_key,
        }

    def _set_channel(self):
        """Prompts the user for a new value of self._channel1."""
        current_channel = self._window.display_settings.image_channel
        if self._measurement_channel is not None:
            current_channel = self._measurement_channel
        channel_count = len(self._find_available_channels())

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            self._measurement_channel = ImageChannel(index_zero=new_channel_index - 1)
            self.refresh_data()

    def _set_nucleus_radius(self):
        """Prompts the user for a new nucleus radius."""
        new_radius = dialog.prompt_float("Nucleus radius",
                                         "What radius (in μm) around the center position would you like to use?",
                                         minimum=0.01, default=self._nucleus_radius_um)
        if new_radius is not None:
            self._nucleus_radius_um = new_radius
            self.refresh_data()  # Redraws the circles

    def _set_intensity_key(self):
        """Prompts the user for a new intensity key."""
        new_key = dialog.prompt_str("Storage key",
                                    "Under what key should the intensities be stored?"
                                    "\nYou can choose a different value than the default if you want"
                                    " to maintain different sets of intensities.",
                                    default=self._intensity_key)
        if new_key is not None and len(new_key) > 0:
            self._intensity_key = new_key

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if dt != 0:
            return True

        intensity_color = (1, 1, 0, 0.5)

        resolution = self._experiment.images.resolution()
        if dz != 0:
            return True  # Don't draw at this Z

        diameter_x_px = 2 * self._nucleus_radius_um / resolution.pixel_size_x_um
        diameter_y_px = 2 * self._nucleus_radius_um / resolution.pixel_size_y_um
        if abs(dz) <= 3:
            self._ax.add_artist(Ellipse((position.x, position.y), width=diameter_x_px, height=diameter_y_px,
                                        fill=True, facecolor=intensity_color))
        return True

    def _find_available_channels(self) -> Set[ImageChannel]:
        """Finds all channels that are available in all open experiments."""
        channels = set()
        for experiment in self._window.get_active_experiments():
            for channel in experiment.images.get_channels():
                channels.add(channel)
        return channels

    def _intensity_key_already_exists(self, key: str) -> bool:
        for experiment in self._window.get_active_experiments():
            if experiment.positions.has_position_data_with_name(key):
                return True
        return False

    def _record_intensities(self):
        channels = self._find_available_channels()
        if self._measurement_channel is None or self._measurement_channel not in channels:
            raise UserError("Invalid first channel", "Please set a channel to measure in"
                                                     " using the Parameters menu.")
        if self._intensity_key_already_exists(self._intensity_key):
            if not dialog.prompt_confirmation("Intensities", "Warning: previous intensities stored under the key "
                                                             "\""+self._intensity_key+"\" will be overwritten.\n\n"
                                                             "This cannot be undone. Do you want to continue?\n\n"
                                                             "If you press Cancel, you can go back and choose a"
                                                             " different key in the Parameters menu."):
                return

        worker_job.submit_job(self._window, _RecordIntensitiesJob(self._nucleus_radius_um, self._measurement_channel,
                                                                  self._intensity_key))
        self.update_status("Started recording all intensities...")

    def _get_figure_title(self) -> str:
        return (f"Intensity measurement (circle)\n"
                f"Time point {self._time_point.time_point_number()}    (z={self._get_figure_title_z_str()}, "
                f"c={self._display_settings.image_channel.index_one})")
