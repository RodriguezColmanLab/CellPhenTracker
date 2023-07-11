"""Uses a simple sphere of a given radius for segmentation"""
import math
import numpy
import random
from typing import Optional, Dict, Any, Tuple, List

import matplotlib.cm
from matplotlib.colors import Colormap, ListedColormap
from numpy import ndarray

from organoid_tracker.core import UserError, bounding_box, TimePoint
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
        "Intensity//Record-Record intensities//Record using vertex model...": lambda: _view_intensities(window)
    }


def _view_intensities(window: Window):
    activate(_SeedSegmentationVisualizer(window))


def _get_intensity(position: Position, intensity_image: Image, mask: Mask) -> Optional[int]:
    mask.center_around(position)
    if mask.count_pixels() == 0:
        return None
    masked_image = mask.create_masked_image(intensity_image)
    return int(masked_image.sum())


def _create_watershed_image(max_radius_um: float, positions: List[Position], original_image: Image, resolution: ImageResolution) -> Image:
    """Creates a watershed image. Any label - 2 corresponds to the index in the positions list."""

    # Create marker image and seeds for distance transform
    marker_image = Image.zeros_like(original_image, dtype=numpy.uint8)
    distance_transform_image = Image(numpy.ones_like(original_image.array, dtype=numpy.uint8), offset=original_image.offset)
    for i, position in enumerate(positions):
        marker_image.set_pixel(position, i + 2)
        distance_transform_image.set_pixel(position, 0)

    # Perform distance transform
    from scipy.ndimage import distance_transform_edt
    watershed_landscape = distance_transform_edt(input=distance_transform_image.array, sampling=resolution.pixel_size_zyx_um)

    # This disables the watershed for all positions too far away
    marker_image.array[watershed_landscape > max_radius_um] = 1

    # Perform watershed
    import mahotas
    result = mahotas.cwatershed(watershed_landscape, marker_image.array)

    return Image(result, offset=original_image.offset)


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

    def compute(self) -> Tuple[Dict[Position, int], Dict[Position, int]]:
        resolution = self._experiment_copy.images.resolution()

        intensities = dict()
        volumes_px3 = dict()
        for time_point in self._experiment_copy.positions.time_points():
            print(f"Working on time point {time_point.time_point_number()}...")
            positions = list(self._experiment_copy.positions.of_time_point(time_point))

            # Load images
            measurement_image_1 = self._experiment_copy.images.get_image(time_point, self._measurement_channel_1)
            measurement_image_2 = None
            if self._measurement_channel_2 is not None:
                measurement_image_2 = self._experiment_copy.images.get_image(time_point, self._measurement_channel_2)
            watershed_image = _create_watershed_image(self._radius_um, positions, measurement_image_1, resolution)

            # Calculate intensities
            for i, position in enumerate(positions):
                cell_values_1 = measurement_image_1.array[watershed_image.array == i + 2]
                intensity = cell_values_1.sum()
                volume_px3 = len(cell_values_1)
                if volume_px3 == 0:
                    continue  # Failed for this cell

                if self._measurement_channel_2 is not None:
                    # Divide by the other channel
                    cell_values_2 = measurement_image_2.array[watershed_image.array == i + 2]
                    intensity /= cell_values_2.sum()

                intensities[position] = int(intensity)
                volumes_px3[position] = int(volume_px3)
        return intensities, volumes_px3

    def on_finished(self, result: Tuple[Dict[Position, int], Dict[Position, int]]):
        intensities, volume_px3 = result

        intensity_calculator.set_raw_intensities(self._experiment_original, intensities, volume_px3)
        dialog.popup_message("Intensities recorded", "All intensities have been recorded.\n\n"
                                                     "Your next step is likely to set a normalization. This can be\n"
                                                     "done from the Intensity menu in the main screen of the program.")


class _SeedSegmentationVisualizer(ExitableImageVisualizer):
    """First, specify the measurement channels and the maximum radius in the Parameters menu.
    Then, if you are happy with the masks, use Edit -> Record intensities."""

    _channel_1: Optional[ImageChannel] = None
    _channel_2: Optional[ImageChannel] = None
    _nucleus_radius_um: float = 6

    _overlay_image: Optional[Image] = None  # 3D image with labels: 1 for cell 1, 2 for the second, etc.
    _label_colormap: Colormap

    def __init__(self, window: Window):
        super().__init__(window)

        # Initialize or random colormap
        source_colormap: Colormap = matplotlib.cm.jet
        samples = [source_colormap(sample_pos / 1000) for sample_pos in range(1000)]
        random.Random("fixed seed to ensure same colors").shuffle(samples)
        samples[0] = (0, 0, 0, 0)  # Force background to black
        samples[1] = (0, 0, 0, 0)  # Force first label to black too, this is also background
        self._label_colormap = ListedColormap(samples)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Channels-Record intensities...": self._record_intensities,
            "Parameters//Channel-Set first channel...": self._set_channel,
            "Parameters//Channel-Set second channel (optional)...": self._set_channel_two,
            "Parameters//Radius-Set maximum nucleus radius...": self._set_max_nucleus_radius,
        }

    def _set_channel(self):
        """Prompts the user for a new value of self._channel1."""
        current_channel = self._window.display_settings.image_channel
        if self._channel_1 is not None:
            current_channel = self._channel_1
        channel_count = len(self._experiment.images.get_channels())

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            self._channel_1 = ImageChannel(index_zero=new_channel_index - 1)
            self.refresh_data()

    def _set_channel_two(self):
        """Prompts the user for a new value of either self._channel2.
        """
        current_channel = self._window.display_settings.image_channel
        if self._channel_2 is not None:
            current_channel = self._channel_2
        channel_count = len(self._experiment.images.get_channels())

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use as the denominator"
                                                                  f" (1-{channel_count}, inclusive)?\n\nIf you don't want to compare two"
                                                                  f" channels, and just want to\nview one channel, set this value to 0.",
                                              minimum=0, maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            if new_channel_index == 0:
                self._channel_2 = None
            else:
                self._channel_2 = ImageChannel(index_zero=new_channel_index -1)
            self.refresh_data()

    def _set_max_nucleus_radius(self):
        """Prompts the user for a new nucleus radius."""
        new_radius = dialog.prompt_float("Nucleus radius",
                                         "What radius (in μm) around the center position would you like to use?",
                                         minimum=0.01, default=self._nucleus_radius_um)
        if new_radius is not None:
            self._nucleus_radius_um = new_radius
            self.refresh_data()  # Redraws the spheres

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

    def _calculate_time_point_metadata(self):
        resolution = self._experiment.images.resolution(allow_incomplete=True)
        positions = list(self._experiment.positions.of_time_point(self._time_point))

        if self._channel_1 is None or resolution.is_incomplete() or len(positions) == 0:
            # Not all information is present
            self._overlay_image = None
            return

        image_size = self._experiment.images.image_loader().get_image_size_zyx()
        original_image = None
        if image_size is not None:
            # We know an image size, we can construct a fake image (doesn't matter for the result)
            offset = self._experiment.images.offsets.of_time_point(self._time_point)
            original_image = Image(numpy.zeros(image_size, dtype=numpy.uint8), offset=offset)
        if original_image is None:
            # We don't know the image size, load the image
            original_image = self._experiment.images.get_image(self._time_point, self._channel_1)
        if original_image is None:
            # No image was found, fail
            self._overlay_image = None
            return

        self._overlay_image = _create_watershed_image(self._nucleus_radius_um, positions, original_image, resolution)

    def should_show_image_reconstruction(self) -> bool:
        return self._overlay_image is not None

    def reconstruct_image(self, time_point: TimePoint, z: int, rgb_canvas_2d: ndarray):
        """Draws the labels in color to the rgb image."""
        if self._overlay_image is None:
            return
        offset = self._overlay_image.offset  # Will be the same as the time point image offset
        image_z = int(z - offset.z)
        if image_z < 0 or image_z > self._overlay_image.array.shape[0]:
            return   # Nothing to draw here

        labels: ndarray = self._overlay_image.array[image_z]
        colored: ndarray = self._label_colormap(labels.flatten())
        colored = colored.reshape((rgb_canvas_2d.shape[0], rgb_canvas_2d.shape[1], 4))
        rgb_canvas_2d[:, :, :] += colored[:, :, 0:3]
        rgb_canvas_2d.clip(min=0, max=1, out=rgb_canvas_2d)

    def reconstruct_image_3d(self, time_point: TimePoint, rgb_canvas_3d: ndarray):
        """Draws the labels in color to the rgb image."""
        if self._overlay_image is None:
            return
        colored: ndarray = self._label_colormap(self._overlay_image.array.flatten())
        colored = colored.reshape((rgb_canvas_3d.shape[0], rgb_canvas_3d.shape[1], rgb_canvas_3d.shape[2], 4))
        rgb_canvas_3d[:, :, :, :] += colored[:, :, :, 0:3]
        rgb_canvas_3d.clip(min=0, max=1, out=rgb_canvas_3d)
