"""Uses a watershed segmentation on membranes for segmentation"""
import os
from typing import Optional, Dict, Any, NamedTuple, List, Tuple

import cv2
import mahotas
import matplotlib.colors
import matplotlib.cm
import numpy
import skimage.morphology
import tifffile
from numpy import ndarray

from organoid_tracker.core import UserError, TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


# Cells that are larger than 2x the median size, or smaller than the median / 2, are ignored
_MAX_FACTOR_FROM_MEDIAN = 2



def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Record intensities//Record intensity using membranes...": lambda: _view_intensities(window)
    }


def _view_intensities(window: Window):
    activate(_NucleusSegmentationVisualizer(window))


def _is_out_of_bounds(coords: Tuple[int, ...], image: ndarray) -> bool:
    """Checks if the given coord is out of bounds for the given array."""
    for i in range(len(coords)):
        if coords[i] < 0 or coords[i] >= image.shape[i]:
            return True
    return False


def _get_organoid_mask(membrane_image: ndarray) -> ndarray:
    # Threshold the membrane, fill the holes
    membrane_mask_image = membrane_image > membrane_image.max() / 10
    for z in range(membrane_mask_image.shape[0]):
        membrane_mask_image[z] = mahotas.close_holes(membrane_mask_image[z])

    # Remove everything but the largest object
    membrane_mask_image, _ = mahotas.label(membrane_mask_image)
    sizes = mahotas.labeled.labeled_size(membrane_mask_image)
    largest_object = int(numpy.argmax(sizes[1:]) + 1)
    membrane_mask_image = (membrane_mask_image == largest_object).astype(numpy.uint8)

    # Make the mask a bit smaller
    for z in range(membrane_mask_image.shape[0]):
        membrane_mask_image[z] = skimage.morphology.erosion(membrane_mask_image[z], footprint=numpy.ones((8, 8)))

    return membrane_mask_image


class _Result(NamedTuple):
    """Holds the resulting volumes and intensities."""
    volumes: Dict[Position, int]
    intensities: Dict[Position, int]

    @staticmethod
    def create() -> "_Result":
        """Creates an empty result, so that you can fill it with your measurements."""
        return _Result(volumes=dict(), intensities=dict())


def _create_watershed_image(membrane_image: ndarray, positions: List[Position], image_offset: Position, resolution: ImageResolution) -> ndarray:
    membrane_image = membrane_image.astype(numpy.float32)  # We don't want to modify the original array in this function
    membrane_image /= membrane_image.max()

    # Create positions image, from which we start the watershed
    positions_image = numpy.zeros(membrane_image.shape, dtype=numpy.uint16)  # Small numbered squares at positions
    hole_at_positions_image = numpy.ones_like(membrane_image, dtype=numpy.uint8)  # 1 everywhere, except at positions
    for i, position in enumerate(positions):
        x = int(position.x - image_offset.x)
        y = int(position.y - image_offset.y)
        z = int(position.z - image_offset.z)
        if _is_out_of_bounds((z, y, x), positions_image):
            continue  # Outside image
        positions_image[z - 1: z+1, y - 4:y + 4, x - 4:x + 4] = i + 1
        hole_at_positions_image[z, y, x] = 0

    # Do the distance transform (from 0 to 1)
    from scipy.ndimage import distance_transform_edt
    distance_landscape = distance_transform_edt(input=hole_at_positions_image,
                                                sampling=resolution.pixel_size_zyx_um)
    max_radius_um = 7
    numpy.clip(distance_landscape, a_min=0, a_max=max_radius_um, out=distance_landscape)
    distance_landscape /= (max_radius_um * 3)  # So goes from 0 to 0.33

    # Do not watershed above the distance threshold
    outside_organoid_value = positions_image.max() + 1
    positions_image[_get_organoid_mask(membrane_image) == 0] = outside_organoid_value

    # Watershed!
    watershed_image: ndarray = mahotas.cwatershed(distance_landscape + membrane_image, markers=positions_image)
    watershed_image[watershed_image == outside_organoid_value] = 0
    return watershed_image.astype(numpy.int16)


class _ExportSegmentationTask(Task):
    _experiment_copy: Experiment
    _membrane_channel: ImageChannel
    _output_folder: str

    def __init__(self, experiment: Experiment, membrane_channel: ImageChannel, output_folder: str):
        experiment.links.sort_tracks_by_x()
        self._experiment_copy = experiment.copy_selected(images=True, positions=True, links=True)
        self._membrane_channel = membrane_channel
        self._output_folder = output_folder

    def compute(self) -> int:
        os.makedirs(self._output_folder, exist_ok=True)
        resolution = self._experiment_copy.images.resolution()
        for time_point in self._experiment_copy.positions.time_points():
            print(f"Working on time point {time_point.time_point_number()}...")

            # Do the watershed
            membrane_image = self._experiment_copy.images.get_image_stack(time_point, self._membrane_channel)
            positions = list(self._experiment_copy.positions.of_time_point(time_point))
            if membrane_image is None or len(positions) == 0:
                continue  # Cannot do watershed here
            image_offset = self._experiment_copy.images.offsets.of_time_point(time_point)
            watershed_image = _create_watershed_image(membrane_image, positions, image_offset, resolution)

            # Post-processing (remove background and large/small segmentations)
            watershed_image[watershed_image == watershed_image.max()] = 0
            sizes = mahotas.labeled.labeled_size(watershed_image)
            median_volume = numpy.median(sizes)
            for i, size in enumerate(sizes):
                if i == 0:
                    continue
                if size < median_volume / _MAX_FACTOR_FROM_MEDIAN or size > median_volume * _MAX_FACTOR_FROM_MEDIAN:
                    watershed_image[watershed_image == i] = watershed_image.max() + 1

            # Make the image indexed by track_id
            if self._experiment_copy.links.has_links():
                watershed_image_correct_colors = numpy.zeros_like(watershed_image, dtype=numpy.uint16)

                # Outline organoid mask
                watershed_image_correct_colors[7:][watershed_image[7:] > 0] = 1

                # Add back ids
                for id, track in self._experiment_copy.links.find_all_tracks_and_ids():
                    if time_point.time_point_number() >= track.min_time_point_number() and time_point.time_point_number() <= track.max_time_point_number():
                        position = track.find_position_at_time_point_number(time_point.time_point_number())
                        try:
                            old_index = positions.index(position) + 1
                            watershed_image_correct_colors[watershed_image == old_index] = id + 2
                        except IndexError:
                            pass  # Position is not in experiment.positions for whatever reason
            else:
                watershed_image_correct_colors = watershed_image.astype(numpy.int16)
                # Make background use index 0
                watershed_image[watershed_image_correct_colors == watershed_image_correct_colors.max()] = 0

            del watershed_image

            # Save the image
            tifffile.imwrite(os.path.join(self._output_folder, f"image_t{time_point.time_point_number()}.tif"), watershed_image_correct_colors, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})
        return 0

    def on_finished(self, result: int):
        dialog.popup_message("Intensities recorded", "All intensities have been recorded.\n\n"
                                                     "Your next step is likely to set a normalization. This can be\n"
                                                     "done from the Intensity menu in the main screen of the program.")

class _RecordIntensitiesTask(Task):
    """Records the intensities of all positions."""

    _experiment_original: Experiment
    _experiment_copy: Experiment
    _measurement_channel_1: ImageChannel
    _measurement_channel_2: Optional[ImageChannel]
    _membrane_channel: ImageChannel

    def __init__(self, experiment: Experiment, membrane_channel: ImageChannel,
                 measurement_channel_1: ImageChannel, measurement_channel_2: Optional[ImageChannel]):
        # Make copy of experiment - so that we can safely work on it in another thread
        self._experiment_original = experiment
        self._experiment_copy = experiment.copy_selected(images=True, positions=True)
        self._membrane_channel = membrane_channel
        self._measurement_channel_1 = measurement_channel_1
        self._measurement_channel_2 = measurement_channel_2

    def compute(self) -> _Result:
        resolution = self._experiment_copy.images.resolution()

        results = _Result.create()
        for time_point in self._experiment_copy.positions.time_points():
            print(f"Working on time point {time_point.time_point_number()}...")

            # Load images
            measurement_image_1 = self._experiment_copy.images.get_image(time_point, self._measurement_channel_1)
            if measurement_image_1 is None:
                continue  # No image for this time point
            measurement_image_2 = None
            if self._measurement_channel_2 is not None:
                measurement_image_2 = self._experiment_copy.images.get_image(time_point, self._measurement_channel_2)
                if measurement_image_2 is None:
                    continue  # No image for this time point

            # Load images
            membrane_image = self._experiment_copy.images.get_image_stack(time_point, self._membrane_channel)
            positions = list(self._experiment_copy.positions.of_time_point(time_point))
            image_offset = self._experiment_copy.images.offsets.of_time_point(time_point)
            if membrane_image is None or len(positions) == 0:
                continue  # Cannot do watershed here

            watershed_image = _create_watershed_image(membrane_image, positions, image_offset, resolution)
            median_volume = numpy.median(mahotas.labeled.labeled_size(watershed_image))

            # Record intensities and volumes
            for i, position in enumerate(positions):
                measurement_1 = measurement_image_1.array[watershed_image == i + 1]
                volume = len(measurement_1)
                if volume == 0 or volume < median_volume / _MAX_FACTOR_FROM_MEDIAN \
                        or volume > median_volume * _MAX_FACTOR_FROM_MEDIAN:
                    continue  # Volume too big or too small, or not found at all
                if measurement_image_2 is not None:
                    measurement_2 = measurement_image_2.array[watershed_image == i + 1]
                    intensity = measurement_1.sum() / measurement_2.sum()
                else:
                    intensity = measurement_1.sum()
                results.intensities[position] = int(intensity)
                results.volumes[position] = int(volume)

        return results

    def on_finished(self, result: _Result):
        intensity_calculator.set_raw_intensities(self._experiment_original, result.intensities, result.volumes)
        dialog.popup_message("Intensities recorded", "All intensities have been recorded.\n\n"
                                                     "Your next step is likely to set a normalization. This can be\n"
                                                     "done from the Intensity menu in the main screen of the program.")


class _NucleusSegmentationVisualizer(ExitableImageVisualizer):
    """First, specify the membrane and measurement channels in the Parameters menu.
    Then, record the intensities of each cell. If you are happy with the masks, then
    use Edit -> Record intensities."""

    _channel_1: Optional[ImageChannel] = None
    _channel_2: Optional[ImageChannel] = None
    _membrane_channel: Optional[ImageChannel] = None

    _overlay_image: Optional[Image] = None  # Used to show the segmentation
    _label_colormap: matplotlib.colors.Colormap

    def __init__(self, window: Window):
        super().__init__(window)

        # Create a random colormap
        values = numpy.linspace(0, 1, 512)
        numpy.random.shuffle(values)
        colors = matplotlib.cm.jet(values)
        colors[0] = (0, 0, 0, 1)
        self._label_colormap = matplotlib.colors.ListedColormap(colors)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Channels-Record intensities...": self._record_intensities,
            "Edit//Channels-Export segmentation images...": self._export_segmentation_images,
            "Parameters//Channel-Set first channel...": self._set_channel,
            "Parameters//Channel-Set second channel (optional)...": self._set_channel_two,
            "Parameters//Nuclei-Set membrane channel...": self._set_membrane_channel,
        }

    def _set_channel(self):
        """Prompts the user for a new value of self._channel1."""
        channels = self._experiment.images.get_channels()
        current_channel = self._channel_1.index_zero if self._channel_1 is not None else \
            self._window.display_settings.image_channel.index_zero
        channel_count = len(channels)

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel + 1)
        if new_channel_index is not None:
            self._channel_1 = ImageChannel(index_zero=new_channel_index - 1)
            self.refresh_data()

    def _set_channel_two(self):
        """Prompts the user for a new value of either self._channel2.
        """
        channels = self._experiment.images.get_channels()
        current_channel = self._channel_2.index_zero if self._channel_2 is not None else\
            self._window.display_settings.image_channel.index_zero
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
                self._channel_2 = ImageChannel(index_zero=new_channel_index - 1)
            self.refresh_data()

    def _set_membrane_channel(self):
        """Prompts the user for a new membrane channel."""
        channels = self._experiment.images.get_channels()
        current_channel = self._membrane_channel.index_zero if self._membrane_channel is not None else \
            self._window.display_settings.image_channel.index_zero
        channel_count = len(channels)

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel contain the membranes"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel + 1)
        if new_channel_index is not None:
            self._membrane_channel = ImageChannel(index_zero=new_channel_index - 1)
            self.refresh_data()

    def _record_intensities(self):
        channels = self._experiment.images.get_channels()
        if self._membrane_channel is None or self._membrane_channel not in channels:
            raise UserError("Invalid membrane channel", "Please set which channel contains the membranes, as this channel"
                                                       " is used for the watershed segmentation.")
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
            _RecordIntensitiesTask(self._experiment, self._membrane_channel, self._channel_1,
                                   self._channel_2))
        self.update_status("Started recording all intensities...")

    def _export_segmentation_images(self):
        channels = self._experiment.images.get_channels()
        if self._membrane_channel is None or self._membrane_channel not in channels:
            raise UserError("Invalid membrane channel",
                            "Please set which channel contains the membranes, as this channel"
                            " is used for the watershed segmentation.")
        if not dialog.prompt_confirmation("Segmentation", "This will write out segmentation images, based on the"
                                                          " membrane signal. It will not record any intensities."):
            return

        output_folder = dialog.prompt_save_file("Segmentation folder", [("Folder", "*")])
        self._window.get_scheduler().add_task(
            _ExportSegmentationTask(self._experiment, self._membrane_channel, output_folder))
        self.update_status("Started writing the segmentations...")

    def _calculate_time_point_metadata(self):
        resolution = self._experiment.images.resolution(allow_incomplete=True)
        positions = list(self._experiment.positions.of_time_point(self._time_point))

        if resolution.is_incomplete() or len(positions) == 0 or self._membrane_channel is None:
            # Not all information is present
            self._overlay_image = None
            return

        membrane_image = self._experiment.images.get_image(self._time_point, self._membrane_channel)
        if membrane_image is None or len(positions) == 0:
            self._overlay_image = None
            return

        image_offset = self._experiment.images.offsets.of_time_point(self._time_point)
        watershed_image = _create_watershed_image(membrane_image.array, positions, image_offset, resolution)
        median_volume = numpy.median(mahotas.labeled.labeled_size(watershed_image))
        watershed_image, highest_label = mahotas.labeled.filter_labeled(watershed_image, min_size=median_volume / _MAX_FACTOR_FROM_MEDIAN,
                                                                        max_size=median_volume * _MAX_FACTOR_FROM_MEDIAN)
        watershed_image[mahotas.labeled.borders(watershed_image)] = 0
        self._overlay_image = Image(watershed_image, image_offset)

    def should_show_image_reconstruction(self) -> bool:
        return self._overlay_image is not None

    def reconstruct_image(self, time_point: TimePoint, z: int, rgb_canvas_2d: ndarray):
        """Draws the labels in color to the rgb image."""
        if self._overlay_image is None:
            return
        offset = self._overlay_image.offset  # Will be the same as the time point image offset
        image_z = int(z - offset.z)
        if image_z < 0 or image_z > self._overlay_image.array.shape[0]:
            return  # Nothing to draw here

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
