from typing import Any, Dict

import numpy
import scipy
import skimage
from matplotlib.colors import Colormap, LinearSegmentedColormap
from numpy import ndarray

from organoid_tracker.core import TimePoint, image_coloring
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.gui import option_choose_dialog, dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Intensity//View-View ratiometric image...": lambda: _view_intensities(window)
    }


def _view_intensities(window: Window):
    # First register the colormap if it's not available
    activate(_RatiometricImageViewer(window))


def _create_segmentation_image_2d(image_primary: ndarray, image_secondary: ndarray) -> ndarray:
    """Does a simple foreground/background segmentation based on the primary and secondary image."""
    background_image = image_primary + image_secondary
    background_image_smoothed = scipy.ndimage.gaussian_filter(background_image, sigma=0.9)
    threshold = skimage.filters.threshold_otsu(background_image_smoothed)
    bright_region_mask = background_image_smoothed > threshold * 0.8
    bright_region_mask = skimage.morphology.remove_small_objects(bright_region_mask, min_size=150)
    return bright_region_mask


def _create_segmentation_image_3d(image_primary: ndarray, image_secondary: ndarray) -> ndarray:
    """For consistency with the 2D viewer, we do segmentation layer by layer."""
    resulting_mask = numpy.zeros_like(image_primary, dtype=bool)
    for z in range(image_primary.shape[0]):
        resulting_mask[z] = _create_segmentation_image_2d(image_primary[z], image_secondary[z])
    return resulting_mask


def _create_colormap() -> Colormap:
    """Creates the colormap used for the ratiometric image. Inspired by the Fire colormap in ImageJ."""
    fire_colors = ["#000000", "#5B00D5", "#BE0067", "#FD6B00", "#FFBF00", "#FFFF90", "#FFFFFF"]
    cmap = LinearSegmentedColormap.from_list("ratiometric_fire", fire_colors, N=256)
    cmap.set_bad('black', 1.)
    return cmap


class _RatiometricImageViewer(ExitableImageVisualizer):
    """Use the 'Ratiometric image' menu to set the colorscale, and which channels to divide. Note that this screen is
    purely for display purposes, the settings here have no influence on the measurements."""

    _segmentation_channel: ImageChannel | None = None  # When None, we do thresholding on primary and secondary image
    _primary_channel: ImageChannel | None = ImageChannel(index_one=1)
    _secondary_channel: ImageChannel | None = ImageChannel(index_one=2)

    _vmin: float = 0.1
    _vmax: float = 5.0

    _blur_radius_px: float = 0.9

    _cmap: Colormap

    def __init__(self, window: Window):
        super().__init__(window)

        self._cmap = _create_colormap()

        # Try to auto-set segmentation channel
        for image_channel in self._experiment.images.get_channels():
            colormap = self._experiment.images.get_channel_description(image_channel).colormap
            if colormap.name == image_coloring.SEGMENTATION_COLORMAP_NAME:
                self._segmentation_channel = image_channel
                break

    def _get_color_map(self) -> Colormap:
        return self._cmap

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Ratiometric image//Channels-Set first ratiometric channel...": self._set_primary_channel,
            "Ratiometric image//Channels-Set second ratiometric channel...": self._set_secondary_channel,
            "Ratiometric image//Channels-Set segmentation channel (optional)...": self._set_segmentation_channel,
            "Ratiometric image//Intensity-Set minimum ratio...": self._set_minimum_ratio,
            "Ratiometric image//Intensity-Set maximum ratio...": self._set_maximum_ratio
        }

    def _get_channel_display_names(self) -> list[str]:
        """Gets a list of channel names for display purposes. Matches the channel order from images.get_channels()."""
        return_list = list()
        for channel in self._experiment.images.get_channels():
            channel_name = self._experiment.images.get_channel_description(channel).channel_name
            channel_display_name = str(channel.index_one)
            if channel_display_name != channel_name:
                channel_display_name += " / '" + channel_name + "'"
            return_list.append(channel_display_name)
        return return_list

    def _get_figure_title_channel_str(self) -> str:
        if self._primary_channel is None or self._secondary_channel is None:
            return "none"
        return f"{self._primary_channel.index_one} over {self._secondary_channel.index_one}"

    def _get_status(self) -> tuple[str, bool]:
        if self._primary_channel is None and self._secondary_channel is None:
            return "Use the 'Ratiometric image' menu to set which channels to display.", False
        if self._secondary_channel is None:
            return "Image will be displayed if you also set a secondary channel.", False
        if self._primary_channel is None:
            return "Image will be displayed if you also set a primary channel.", False
        if self._primary_channel == self._secondary_channel:
            return "Image will be displayed once both channels are different.", False

        channel_count = len(self._experiment.images.get_channels())
        if self._primary_channel.index_zero >= channel_count:
            return "The primary channel is out of range, please select a different one.", False
        if self._secondary_channel.index_zero >= channel_count:
            return "The secondary channel is out of range, please select a different one.", False

        if self._segmentation_channel is not None:
            if self._segmentation_channel.index_zero >= channel_count:
                return "The segmentation channel is out of range, please select a different one.", False
            return "Image is now displayed with chosen segmentation.", True

        return "Image is now displayed with Otsu segmentation. You can use another segmentation in the 'Ratiometric image' menu.", True

    def _get_status_message(self) -> str:
        return self._get_status()[0]

    def _set_primary_channel(self):
        display_names = self._get_channel_display_names()
        result = option_choose_dialog.prompt_list("Primary channel", "The channel will be used as the numerator", "Channel", display_names)
        if result is None:
            return
        self._primary_channel = ImageChannel(index_zero=result)
        self.refresh_all()
        self.update_status(f"Primary channel set to {display_names[result]}. {self._get_status_message()}")

    def _set_secondary_channel(self):
        display_names = self._get_channel_display_names()
        result = option_choose_dialog.prompt_list("Secondary channel", "The channel will be used as the denominator", "Channel", display_names)
        if result is None:
            return
        self._secondary_channel = ImageChannel(index_zero=result)
        self.refresh_all()
        self.update_status(f"Secondary channel set to {display_names[result]}. {self._get_status_message()}")

    def _set_segmentation_channel(self):
        display_names = self._get_channel_display_names()
        display_names += "<Otsu segmentation>"
        result = option_choose_dialog.prompt_list("Segmentation channel", "Used to define where the ratio is displayed.", "Channel", display_names)
        if result is None:
            return

        if result == len(display_names) - 1:
            self._segmentation_channel = None  # Use Otsu segmentation
        else:
            self._segmentation_channel = ImageChannel(index_zero=result)
        self.refresh_all()
        self.update_status(f"Segmentation channel set to {display_names[result]}. {self._get_status_message()}")

    def _set_minimum_ratio(self):
        new_ratio = dialog.prompt_float("Minimum ratio", "What should the minimum displayed ratio be?", minimum=0, decimals=2, default=self._vmin)
        if new_ratio is None:
            return
        self._vmin = new_ratio
        self.refresh_all()
        self.update_status(f"Minimum ratio set to {self._vmin:.2f}. {self._get_status_message()}")

    def _set_maximum_ratio(self):
        new_ratio = dialog.prompt_float("Maximum ratio", "What should the maximum displayed ratio be?", minimum=0, decimals=2, default=self._vmax)
        if new_ratio is None:
            return
        self._vmax = new_ratio
        self.refresh_all()
        self.update_status(f"Maximum ratio set to {self._vmax:.2f}. {self._get_status_message()}")

    def _return_2d_image(self, time_point: TimePoint, z: int, channel: ImageChannel, show_next_time_point: bool) -> ndarray | None:
        """We load both images, segment them or load the segmentation, and divide them."""

        # Load both channels
        if self._primary_channel is None or self._secondary_channel is None:
            return None
        if self._primary_channel == self._secondary_channel:
            return None
        image_primary = self._experiment.images.get_image_slice_2d(time_point, self._primary_channel, z)
        image_secondary = self._experiment.images.get_image_slice_2d(time_point, self._secondary_channel, z)
        if image_primary is None or image_secondary is None:
            return None
        if image_primary.shape != image_secondary.shape:
            return None
        image_primary = image_primary.astype(numpy.float32)
        image_secondary = image_secondary.astype(numpy.float32)

        # Load or create segmentation
        if self._segmentation_channel is None:
            image_segmentation = _create_segmentation_image_2d(image_primary, image_secondary)
        else:
            image_segmentation = self._experiment.images.get_image_slice_2d(time_point, self._segmentation_channel, z)
            if image_segmentation is None or image_segmentation.shape != image_primary.shape:
                return None

        image_primary = scipy.ndimage.gaussian_filter(image_primary, sigma=self._blur_radius_px)
        image_secondary = scipy.ndimage.gaussian_filter(image_secondary, sigma=self._blur_radius_px)

        ratiometric_image = numpy.true_divide(image_primary, image_secondary)
        ratiometric_image[~numpy.isfinite(ratiometric_image)] = 0  # Set inf and NaN to 0
        ratiometric_image[image_segmentation == 0] = numpy.nan  # Mask out background

        self._clamp_image(ratiometric_image)

        return ratiometric_image

    def _return_3d_image(self, time_point: TimePoint, channel: ImageChannel, show_next_time_point: bool) -> ndarray | None:
        """We load both images, segment them or load the segmentation, and divide them. Same code as 2D case, with
        minimal adaptations."""

        # Load both channels
        if self._primary_channel is None or self._secondary_channel is None:
            return None
        if self._primary_channel == self._secondary_channel:
            return None
        image_primary = self._experiment.images.get_image_stack(time_point, self._primary_channel)
        image_secondary = self._experiment.images.get_image_stack(time_point, self._secondary_channel)
        if image_primary is None or image_secondary is None:
            return None
        if image_primary.shape != image_secondary.shape:
            return None
        image_primary = image_primary.astype(numpy.float32)
        image_secondary = image_secondary.astype(numpy.float32)

        # Load or create segmentation
        if self._segmentation_channel is None:
            image_segmentation = _create_segmentation_image_3d(image_primary, image_secondary)
        else:
            image_segmentation = self._experiment.images.get_image_stack(time_point, self._segmentation_channel)
            if image_segmentation is None or image_segmentation.shape != image_primary.shape:
                return None

        for z in range(image_primary.shape[0]):
            # For consistency with the 2D viewer, we do Gaussian filtering in 2D
            image_primary[z] = scipy.ndimage.gaussian_filter(image_primary[z], sigma=self._blur_radius_px)
            image_secondary[z] = scipy.ndimage.gaussian_filter(image_secondary[z], sigma=self._blur_radius_px)

        ratiometric_image = numpy.true_divide(image_primary, image_secondary)
        ratiometric_image[~numpy.isfinite(ratiometric_image)] = 0  # Set inf and NaN to 0
        ratiometric_image[image_segmentation == 0] = numpy.nan  # Mask out background

        self._clamp_image(ratiometric_image)

        return ratiometric_image

    def _clamp_image(self, image: ndarray):
        """Clamp the image to the vmin and vmax values for better visualization. Modifies the image in-place."""
        image[image < self._vmin] = self._vmin
        image[image > self._vmax] = self._vmax

    def _draw_extra(self):
        """Draw a status message if the image isn't displayed."""
        message, ok = self._get_status()
        if not ok:
            self._ax.text(0.5, 0.5, message, color="yellow", fontsize=14, ha="center", va="center", transform=self._ax.transAxes)