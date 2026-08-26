import os
from typing import Any, Iterable

import numpy
import scipy
import tifffile
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.gui import dialog, worker_job
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelAppendingImageLoader
from organoid_tracker.image_loading.folder_image_loader import FolderImageLoader
from organoid_tracker.imaging import list_io
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Tools//Segment-Segmentation//Segment-Segment with intensity thresholding...": lambda: _segment_foreground_background(window),
    }


def _segment_foreground_background(window: Window):
    activate(_ForegroundBackgroundSegmentationVisualizer(window))


def _get_next_message() -> str:
    return "Use Edit -> Save segmentation to save the segmentation result as a binary image."



class _CachedThreshold:
    """Holds the absolute threshold value for a specific time point and channel. This is used to avoid
    recalculating the threshold for the same time point and channel multiple times."""

    _threshold_abs: float| None = None

    _cached_for_time_point: TimePoint | None = None
    _cached_for_channel: ImageChannel | None = None

    def get_value(self, time_point: TimePoint, channel: ImageChannel) -> float | None:
        if time_point == self._cached_for_time_point and channel == self._cached_for_channel:
            return self._threshold_abs
        return None

    def set_value(self, time_point: TimePoint, channel: ImageChannel, value: float):
        self._cached_for_time_point = time_point
        self._cached_for_channel = channel
        self._threshold_abs = value

    def reset(self):
        self._cached_for_time_point = None
        self._cached_for_channel = None
        self._threshold_abs = None


class _Thresholder:

    blur_radius_px: int = 2
    erosion_dilation_radius_px: int = 0
    relative_threshold: float = 0.25
    threshold_quantile: float = 0.99

    def copy(self) -> "_Thresholder":
        new_thresholder = _Thresholder()
        new_thresholder.blur_radius_px = self.blur_radius_px
        new_thresholder.erosion_dilation_radius_px = self.erosion_dilation_radius_px
        new_thresholder.relative_threshold = self.relative_threshold
        new_thresholder.threshold_quantile = self.threshold_quantile
        return new_thresholder

    def calculate_abs_threshold(self, image_3d: ndarray) -> float:
        return numpy.quantile(image_3d, self.threshold_quantile) * self.relative_threshold

    def threshold(self, image_2d: ndarray, threshold_abs: float) -> ndarray:

        # Apply Gaussian blur if specified
        if self.blur_radius_px > 0:
            image_blurred = scipy.ndimage.gaussian_filter(image_2d, sigma=self.blur_radius_px)
        else:
            image_blurred = image_2d

        thresholded_image = image_blurred > threshold_abs

        # Apply erosion or dilation if specified
        if self.erosion_dilation_radius_px != 0:
            struct = scipy.ndimage.generate_binary_structure(2, 1)
            struct = scipy.ndimage.iterate_structure(struct, abs(self.erosion_dilation_radius_px))
            if self.erosion_dilation_radius_px > 0:
                thresholded_image = scipy.ndimage.binary_dilation(thresholded_image, structure=struct)
            else:
                thresholded_image = scipy.ndimage.binary_erosion(thresholded_image, structure=struct)

        return thresholded_image


class _SegmentationExport(WorkerJob):

    _thresholder: _Thresholder
    _channel: ImageChannel
    _export_folder: str
    _multiple_experiments: bool

    _list_file: str

    def __init__(self, thresholder: _Thresholder, channel: ImageChannel, export_folder: str, multiple_experiments: bool):
        self._thresholder = thresholder
        self._channel = channel
        self._export_folder = export_folder
        self._multiple_experiments = multiple_experiments

        self._list_file = os.path.join(self._export_folder, "Segmentation export" + list_io.FILES_LIST_EXTENSION)
        if os.path.exists(self._list_file):
            os.remove(self._list_file)

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        experiment_copy = experiment.copy_selected(images=True, name=True)
        experiment_copy.last_save_file = experiment.last_save_file
        return experiment_copy

    def gather_data(self, experiment_copy: Experiment) -> Any:
        output_folder = self._export_folder if not self._multiple_experiments else \
            os.path.join(self._export_folder, experiment_copy.name.get_save_name())
        os.makedirs(output_folder, exist_ok=True)
        for time_point in self.reporting_progress(experiment_copy.images.time_points()):
            image_stack = experiment_copy.images.get_image_stack(time_point, self._channel)
            if image_stack is None:
                continue
            threshold = self._thresholder.calculate_abs_threshold(image_stack)
            thresholded = numpy.zeros_like(image_stack, dtype=numpy.uint8)
            for z in range(image_stack.shape[0]):
                thresholded[z] = self._thresholder.threshold(image_stack[z], threshold)
            output_file = os.path.join(output_folder, f"thresholded_t{time_point.time_point_number()}.tif")
            tifffile.imwrite(output_file, scipy.ndimage.label(thresholded), compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

        min_time_point_number = experiment_copy.images.first_time_point_number()
        max_time_point_number = experiment_copy.images.last_time_point_number()
        if min_time_point_number is not None and max_time_point_number is not None:
            # Register the new thresholded images as a new channel
            tiff_image_loader = FolderImageLoader(output_folder, "thresholded_t{time}.tif", min_time_point_number, max_time_point_number, 1, 1)
            combined_image_loader = ChannelAppendingImageLoader([experiment_copy.images.image_loader(), tiff_image_loader])
            experiment_copy.images.image_loader(combined_image_loader)

            # Append this experiment to the list file, so that it can be opened later
            list_io.save_experiment_list_file([experiment_copy], self._list_file, append_to_file=True)

        return "Done"  # Any value works, as long as it truthy


    def on_finished(self, data: Iterable[Any]):
        result = dialog.prompt_options("Segmentation export finished", "Segmentation export finished successfully.",
                              option_1="OK",
                              option_2="Open export folder")
        if result == 2:
            dialog.open_file(self._export_folder)


class _ForegroundBackgroundSegmentationVisualizer(ExitableImageVisualizer):
    """For classical intensity-based segmentation. Use the Parameters menu to set the blur radius and absolute
    threshold for segmentation. Pixels with intensity above the threshold will be considered foreground.
    Use Edit -> Save segmentation to save the segmentation result as a labeled image."""

    _thresholder: _Thresholder
    _cached_threshold_abs: float= 0

    def __init__(self, window: Window):
        super().__init__(window)
        self._thresholder = _Thresholder()

    def should_show_image_reconstruction(self) -> bool:
        return True

    def _calculate_time_point_metadata(self):
        time_point = self._display_settings.time_point
        channel = self._display_settings.image_channel
        image_stack = self._experiment.images.get_image_stack(time_point, channel)
        if image_stack is None:
            self._cached_threshold_abs = 0
            return

        # Calculate the absolute threshold based on the quantile and multiplier
        threshold_value = self._thresholder.calculate_abs_threshold(image_stack)
        self._cached_threshold_abs = threshold_value

    def get_extra_menu_options(self) -> dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Parameters//Set blur radius...": self._set_blur_radius,
            "Parameters//Set relative threshold...": self._set_relative_threshold,
            "Parameters//Set erosion/dilation radius...": self._set_erosion_dilation_radius,
            "Edit//Segment-Save segmentation...": self._export_segmentation
        }

    def _set_relative_threshold(self):
        percentile = int(self._thresholder.threshold_quantile * 100)
        threshold = dialog.prompt_float("Relative threshold", "Set the relative threshold value for segmentation."
                                        f" The max value of the image is defined as the {percentile}th percentile of the image"
                                        f" intensity. The absolute threshold will be calculated as relative_threshold * max_value.",
                                        default=self._thresholder.relative_threshold, minimum=0, maximum=1, decimals=2)
        if threshold is None:
            return
        self._thresholder.relative_threshold = threshold
        self.refresh_all()
        self.update_status(f"Threshold set to {self._thresholder.relative_threshold}. {_get_next_message()}")

    def _set_erosion_dilation_radius(self):
        radius = dialog.prompt_int("Erosion/Dilation radius", "Set the radius for erosion/dilation. "
                                   "Positive values will dilate (expand) the foreground, negative values will erode the foreground. Set to 0 to disable.",
                                   default=self._thresholder.erosion_dilation_radius_px, minimum=-100, maximum=100)
        if radius is None:
            return
        self._thresholder.erosion_dilation_radius_px = radius
        self.refresh_all()
        self.update_status(f"Erosion/dilation radius set to {self._thresholder.erosion_dilation_radius_px} px. {_get_next_message()}")

    def _reconstruct_layer(self, image_2d: ndarray, rgb_canvas_2d: ndarray):
        thresholded_image = self._thresholder.threshold(image_2d, self._cached_threshold_abs)

        image_2d_max = numpy.max(image_2d)
        if image_2d_max <= 0:
            image_2d_max = 1
        image_2d_scaled = image_2d / image_2d_max
        rgb_canvas_2d[:, :, 0] = image_2d_scaled
        rgb_canvas_2d[:, :, 1] = thresholded_image * 0.5 + image_2d_scaled * 0.5
        rgb_canvas_2d[:, :, 2] = thresholded_image * 0.5 + image_2d_scaled * 0.5

    def reconstruct_image(self, time_point: TimePoint, z: int, rgb_canvas_2d: ndarray):
        image = self._experiment.images.get_image_slice_2d(time_point, self._display_settings.image_channel, z)
        if image is None:
            return
        self._reconstruct_layer(image, rgb_canvas_2d)

    def reconstruct_image_3d(self, time_point: TimePoint, rgb_canvas_3d: ndarray):
        image_3d = self._experiment.images.get_image_stack(time_point, self._display_settings.image_channel)
        if image_3d is None:
            return
        for z in range(image_3d.shape[0]):
            print(z, rgb_canvas_3d.shape)
            self._reconstruct_layer(image_3d[z], rgb_canvas_3d[z])

    def _set_blur_radius(self):
        radius = dialog.prompt_int("Blur radius", "By how many pixels should we blur the image? Set to 0 to disable blurring.",
                                   default=self._thresholder.blur_radius_px, minimum=0, maximum=100)
        if radius is None:
            return
        self._thresholder.blur_radius_px = radius
        self.refresh_all()
        self.update_status(f"Blur radius set to {self._thresholder.blur_radius_px} px. {_get_next_message()}")

    def _get_figure_title(self) -> str:
        return (f"Intensity-based segmentation\n"
                f"Time point {self._time_point.time_point_number()}    (z={self._get_figure_title_z_str()}, "
                f"c={self._get_figure_title_channel_str()})")

    def _export_segmentation(self):
        export_folder = dialog.prompt_save_file("Export segmentation", [("Folder", "*")])
        if export_folder is None:
            return

        open_experiments = list(self._window.get_active_experiments())
        job = _SegmentationExport(self._thresholder.copy(), self._display_settings.image_channel, export_folder, len(open_experiments) > 1)
        worker_job.submit_job(self._window, job)
        self.update_status(f"Export started...")
