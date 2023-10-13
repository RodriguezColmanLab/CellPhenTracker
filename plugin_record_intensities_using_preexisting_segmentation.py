"""Uses a simple sphere of a given radius for segmentation"""
import math
import numpy
import random
from typing import Optional, Dict, Any, Tuple, List

import matplotlib.cm
import skimage.measure
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
        "Intensity//Record-Record intensities//Record-Record using pre-existing segmentation...": lambda: _view_intensities(window)
    }


def _view_intensities(window: Window):
    activate(_PreexistingSegmentationVisualizer(window))


def _by_label(region_props: List["skimage.measure._regionprops.RegionProperties"]
              ) -> Dict[int, "skimage.measure._regionprops.RegionProperties"]:
    return_value = dict()
    for region in region_props:
        return_value[region.label] = region
    return return_value


class _RecordIntensitiesTask(Task):
    """Records the intensities of all positions."""

    _experiment_original: Experiment
    _experiment_copy: Experiment
    _segmentation_channel: ImageChannel
    _measurement_channel_1: ImageChannel
    _measurement_channel_2: Optional[ImageChannel]
    _intensity_key: str

    def __init__(self, experiment: Experiment, segmentation_channel: ImageChannel, measurement_channel_1: ImageChannel,
                 measurement_channel_2: Optional[ImageChannel], intensity_key: str):
        # Make copy of experiment - so that we can safely work on it in another thread
        self._experiment_original = experiment
        self._experiment_copy = experiment.copy_selected(images=True, positions=True)
        self._segmentation_channel = segmentation_channel
        self._measurement_channel_1 = measurement_channel_1
        self._measurement_channel_2 = measurement_channel_2
        self._intensity_key = intensity_key

    def compute(self) -> Tuple[Dict[Position, int], Dict[Position, int]]:
        intensities = dict()
        volumes_px3 = dict()
        for time_point in self._experiment_copy.positions.time_points():
            print(f"Working on time point {time_point.time_point_number()}...")

            # Load images
            label_image = self._experiment_copy.images.get_image(time_point, self._segmentation_channel)
            measurement_image_1 = self._experiment_copy.images.get_image(time_point, self._measurement_channel_1)
            measurement_image_2 = None
            if self._measurement_channel_2 is not None:
                measurement_image_2 = self._experiment_copy.images.get_image(time_point, self._measurement_channel_2)
                if measurement_image_2 is None:
                    continue  # Skip this time point, an image is missing

            if label_image is None or measurement_image_1 is None:
                continue  # Skip this time point, an image is missing

            # Calculate intensities
            props_by_label = _by_label(skimage.measure.regionprops(label_image.array))
            for position in self._experiment_copy.positions.of_time_point(time_point):
                index = label_image.value_at(position)
                if index == 0:
                    continue
                props = props_by_label[index]
                intensity = int(numpy.sum(measurement_image_1.array[props.slice] * props.image))
                if measurement_image_2 is not None:
                    intensity_2 = int(numpy.sum(measurement_image_2.array[props.slice] * props.image))
                    intensity /= intensity_2
                intensities[position] = intensity
                volumes_px3[position] = props.area
        return intensities, volumes_px3

    def on_finished(self, result: Tuple[Dict[Position, int], Dict[Position, int]]):
        intensities, volume_px3 = result

        intensity_calculator.set_raw_intensities(self._experiment_original, intensities, volume_px3,
                                                 intensity_key=self._intensity_key)
        dialog.popup_message("Intensities recorded", "All intensities have been recorded.\n\n"
                                                     "Your next step is likely to set a normalization. This can be\n"
                                                     "done from the Intensity menu in the main screen of the program.")


class _PreexistingSegmentationVisualizer(ExitableImageVisualizer):
    """First, specify the segmentation channel (containing a pre-segmented image) and the measurement channel in the
    Parameters menu. Then, use Edit -> Record intensities.

    If you don't have pre-segmented images loaded yet, exit this view and use Edit -> Append image channel.
    """
    _segmented_channel: Optional[ImageChannel] = None
    _channel_1: Optional[ImageChannel] = None
    _channel_2: Optional[ImageChannel] = None
    _intensity_key: str = intensity_calculator.DEFAULT_INTENSITY_KEY
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
            "Parameters//Channel-Set first measurement channel...": self._set_measurement_channel_one,
            "Parameters//Channel-Set second measurement channel (optional)...": self._set_measurement_channel_two,
            "Parameters//Other-Set segmented channel...": self._set_segmented_channel,
            "Parameters//Other-Set storage key...": self._set_intensity_key,
        }

    def _set_intensity_key(self):
        """Prompts the user for a new intensity key."""
        new_key = dialog.prompt_str("Storage key",
                                    "Under what key should the intensities be stored?"
                                    "\nYou can choose a different value than the default if you want"
                                    " to maintain different sets of intensities.",
                                    default=self._intensity_key)
        if new_key is not None and len(new_key) > 0:
            self._intensity_key = new_key

    def _set_segmented_channel(self):
        """Prompts the user for a new value of self._segmentation_channel."""
        channels = self._experiment.images.get_channels()
        current_channel = self._segmented_channel if self._segmented_channel is not None else self._display_settings.image_channel
        channel_count = len(channels)

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            self._segmented_channel = channels[new_channel_index - 1]
            self.refresh_data()

    def _set_measurement_channel_one(self):
        """Prompts the user for a new value of self._channel1."""
        channels = self._experiment.images.get_channels()
        current_channel = self._channel_1 if self._channel_1 is not None else self._display_settings.image_channel
        channel_count = len(channels)

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            self._channel_1 = channels[new_channel_index - 1]
            self.refresh_data()

    def _set_measurement_channel_two(self):
        """Prompts the user for a new value of either self._channel2.
        """
        channels = self._experiment.images.get_channels()
        current_channel = self._channel_2 if self._channel_2 is not None else self._display_settings.image_channel
        channel_count = len(channels)

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use as the denominator"
                                                                  f" (1-{channel_count}, inclusive)?\n\nIf you don't want to compare two"
                                                                  f" channels, and just want to\nview one channel, set this value to 0.",
                                              minimum=0, maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            if new_channel_index == 0:
                self._channel_2 = None
            else:
                self._channel_2 = channels[new_channel_index - 1]
            self.refresh_data()

    def _record_intensities(self):
        channels = self._experiment.images.get_channels()
        if self._segmented_channel is None or self._segmented_channel not in channels:
            raise UserError("Invalid segmentation channel", "Please set a segmentation channel in the Parameters menu.")
        if self._channel_1 is None or self._channel_1 not in channels:
            raise UserError("Invalid first channel", "Please set a measurement channel to measure in"
                                                     " using the Parameters menu.")
        if self._channel_2 is not None and self._channel_2 not in channels:
            raise UserError("Invalid second channel", "The selected second measurement channel is no longer available."
                                                      " Please select a new one in the Parameters menu.")
        if self._experiment.position_data.has_position_data_with_name(self._intensity_key):
            if not dialog.prompt_confirmation("Intensities", "Warning: previous intensities stored under the key "
                                                             "\""+self._intensity_key+"\" will be overwritten.\n\n"
                                                             "This cannot be undone. Do you want to continue?\n\n"
                                                             "If you press Cancel, you can go back and choose a"
                                                             " different key in the Parameters menu."):
                return

        self._window.get_scheduler().add_task(
            _RecordIntensitiesTask(self._experiment, self._segmented_channel, self._channel_1, self._channel_2,
                                   self._intensity_key))
        self.update_status("Started recording all intensities...")

    def should_show_image_reconstruction(self) -> bool:
        if self._segmented_channel is None:
            return False # Nothing to draw
        if self._display_settings.image_channel not in {self._channel_1, self._channel_2, self._segmented_channel}:
            return False # Nothing to draw for this channel
        return True

    def reconstruct_image(self, time_point: TimePoint, z: int, rgb_canvas_2d: ndarray):
        """Draws the labels in color to the rgb image."""
        if self._segmented_channel is None:
            return   # Nothing to draw
        if self._display_settings.image_channel not in {self._channel_1, self._channel_2, self._segmented_channel}:
            return  # Nothing to draw for this channel
        if self._segmented_channel == self._display_settings.image_channel:
            # Avoid drawing on top of the same image
            rgb_canvas_2d[:, :, 0:3] = 0
            if rgb_canvas_2d.shape[-1] == 4:
                rgb_canvas_2d[:, :, 3] = 1  # Also erase alpha channel

        labels = self._experiment.images.get_image_slice_2d(time_point, self._segmented_channel, z)
        if labels is None:
            return  # No image here

        colored: ndarray = self._label_colormap(labels.flatten())
        colored = colored.reshape((rgb_canvas_2d.shape[0], rgb_canvas_2d.shape[1], 4))
        rgb_canvas_2d[:, :, :] += colored[:, :, 0:3]
        rgb_canvas_2d.clip(min=0, max=1, out=rgb_canvas_2d)

    def reconstruct_image_3d(self, time_point: TimePoint, rgb_canvas_3d: ndarray):
        """Draws the labels in color to the rgb image."""
        if self._segmented_channel is None:
            return  # Nothing to draw
        if self._display_settings.image_channel not in {self._channel_1, self._channel_2, self._segmented_channel}:
            return  # Nothing to draw for this channel
        if self._segmented_channel == self._display_settings.image_channel:
            # Avoid drawing on top of the same image
            rgb_canvas_3d[:, :, :, 0:3] = 0
            if rgb_canvas_3d.shape[-1] == 4:
                rgb_canvas_3d[:, :, :, 3] = 1  # Also erase alpha channel

        label_image = self._experiment.images.get_image_stack(self._time_point, self._segmented_channel)
        if label_image is None:
            return  # Nothing to show for this time point
        colored: ndarray = self._label_colormap(label_image.flatten())
        colored = colored.reshape((rgb_canvas_3d.shape[0], rgb_canvas_3d.shape[1], rgb_canvas_3d.shape[2], 4))
        rgb_canvas_3d[:, :, :, :] += colored[:, :, :, 0:3]
        rgb_canvas_3d.clip(min=0, max=1, out=rgb_canvas_3d)
