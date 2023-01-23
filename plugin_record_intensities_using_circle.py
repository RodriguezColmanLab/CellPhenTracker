"""Uses a simple sphere of a given radius for segmentation"""
import math
from typing import Optional, Dict, Any

from matplotlib.patches import Ellipse

from organoid_tracker.core import UserError, bounding_box
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.mask import Mask
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
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


class _RecordIntensitiesTask(Task):
    """Records the intensities of all positions."""

    _experiment_original: Experiment
    _experiment_copy: Experiment
    _radius_um: float
    _measurement_channel_1: ImageChannel
    _measurement_channel_2: Optional[ImageChannel]

    def __init__(self, experiment: Experiment, radius_um: float, measurement_channel_1: ImageChannel, measurement_channel_2: Optional[ImageChannel]):
        # Make copy of experiment - so that we can safely work on it in another thread
        self._experiment_original = experiment
        self._experiment_copy = experiment.copy_selected(images=True, positions=True)
        self._radius_um = radius_um
        self._measurement_channel_1 = measurement_channel_1
        self._measurement_channel_2 = measurement_channel_2

    def compute(self) -> Dict[Position, int]:
        results = dict()
        spherical_mask = _create_circular_mask(self._radius_um, self._experiment_copy.images.resolution())
        for time_point in self._experiment_copy.positions.time_points():

            # Load images
            measurement_image_1 = self._experiment_copy.images.get_image(time_point, self._measurement_channel_1)
            measurement_image_2 = None
            if self._measurement_channel_2 is not None:
                measurement_image_2 = self._experiment_copy.images.get_image(time_point, self._measurement_channel_2)

            # Calculate intensities
            for position in self._experiment_copy.positions.of_time_point(time_point):
                intensity = _get_intensity(position, measurement_image_1, spherical_mask)
                if intensity is not None and self._measurement_channel_2 is not None:
                    intensity_2 = _get_intensity(position, measurement_image_2, spherical_mask)
                    if intensity_2 is None:
                        intensity = None
                    else:
                        intensity /= intensity_2
                if intensity is not None:
                    results[position] = intensity
        return results

    def on_finished(self, result: Dict[Position, int]):
        # Record volumes too, for administrative purposes
        resolution = self._experiment_copy.images.resolution()
        circle_volume_px3 = _create_circular_mask(self._radius_um, resolution).count_pixels()
        volumes = dict(((position, circle_volume_px3) for position in result.keys()))

        intensity_calculator.set_raw_intensities(self._experiment_original, result, volumes)
        dialog.popup_message("Intensities recorded", "All intensities have been recorded.\n\n"
                                                     "Your next step is likely to set a normalization. This can be\n"
                                                     "done from the Intensity menu in the main screen of the program.")


class _CircleSegmentationVisualizer(ExitableImageVisualizer):
    """First, specify the membrane and measurement channels in the Parameters menu.
    Then, record the intensities of each cell. If you are happy with the masks, then
    use Edit -> Record intensities."""

    _channel_1: Optional[ImageChannel] = None
    _channel_2: Optional[ImageChannel] = None
    _nucleus_radius_um: float = 3

    def __init__(self, window: Window):
        super().__init__(window)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Channels-Record intensities...": self._record_intensities,
            "Parameters//Channel-Set first channel...": self._set_channel,
            "Parameters//Channel-Set second channel (optional)...": self._set_channel_two,
            "Parameters//Radius-Set nucleus radius...": self._set_nucleus_radius,
        }

    def _set_channel(self):
        """Prompts the user for a new value of self._channel1."""
        channels = self._experiment.images.get_channels()
        current_channel = 0
        try:
            current_channel = channels.index(self._channel_1)
        except ValueError:
            pass
        channel_count = len(channels)

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel + 1)
        if new_channel_index is not None:
            self._channel_1 = channels[new_channel_index - 1]
            self.refresh_data()

    def _set_channel_two(self):
        """Prompts the user for a new value of either self._channel2.
        """
        channels = self._experiment.images.get_channels()
        current_channel = 0
        try:
            current_channel = channels.index(self._channel_2)
        except ValueError:
            pass
        channel_count = len(channels)

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use as the denominator"
                                                                  f" (1-{channel_count}, inclusive)?\n\nIf you don't want to compare two"
                                                                  f" channels, and just want to\nview one channel, set this value to 0.",
                                              minimum=0, maximum=channel_count,
                                              default=current_channel + 1)
        if new_channel_index is not None:
            if new_channel_index == 0:
                self._channel_2 = None
            else:
                self._channel_2 = channels[new_channel_index - 1]
            self.refresh_data()

    def _set_nucleus_radius(self):
        """Prompts the user for a new nucleus radius."""
        new_radius = dialog.prompt_float("Nucleus radius",
                                         "What radius (in μm) around the center position would you like to use?",
                                         minimum=0.01, default=self._nucleus_radius_um)
        if new_radius is not None:
            self._nucleus_radius_um = new_radius
            self.refresh_data()  # Redraws the circles

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

    def _record_intensities(self):
        channels = self._experiment.images.get_channels()
        if self._channel_1 is None or self._channel_1 not in channels:
            raise UserError("Invalid first channel", "Please set a channel to measure in"
                                                     " using the Parameters menu.")
        if self._channel_2 is not None and self._channel_2 not in channels:
            raise UserError("Invalid second channel", "The selected second channel is no longer available."
                                                      " Please select a new one in the Parameters menu.")
        if not dialog.prompt_confirmation("Intensities", "Warning: previous intensities will be overwritten."
                                                         " This cannot be undone. Do you want to continue?"):
            return

        self._window.get_scheduler().add_task(
            _RecordIntensitiesTask(self._experiment, self._nucleus_radius_um, self._channel_1, self._channel_2))
        self.update_status("Started recording all intensities...")
