from typing import Any, Iterable

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.gui import dialog, worker_job, option_choose_dialog
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Intensity//Record-Background correction//Basic-Correct using pre-existing segmentation...":
            lambda: _normalize_with_preexisting_segmentation(window),
    }


def _find_intensity_keys(window: Window) -> list[str]:
    """Finds all intensity keys that are available in all open experiments."""
    intensity_keys = set()
    for experiment in window.get_active_experiments():
        for key in intensity_calculator.get_regular_intensity_keys(experiment):
            intensity_keys.add(key)
    return sorted(intensity_keys)


def _normalize_with_preexisting_segmentation(window: Window):
    if len(_find_intensity_keys(window)) == 0:
        raise UserError("No intensity keys found", "No intensity keys were found. Please record some intensities first.")
    activate(_BackgroundSubtractionVisualizer(window))



class _RecordBackgroundJob(WorkerJob):

    _measurement_channel: ImageChannel
    _segmentation_channel: ImageChannel
    _intensity_to_correct: str

    def __init__(self, *,  measurement_channel: ImageChannel, segmentation_channel: ImageChannel, intensity_to_correct: str):
        self._measurement_channel = measurement_channel
        self._segmentation_channel = segmentation_channel
        self._intensity_to_correct = intensity_to_correct

    def gather_data(self, experiment_copy: Experiment) -> Any:
        intensity_sum = 0
        intensity_area = 0
        for time_point in self.reporting_progress(experiment_copy.images.time_points()):
            intensity_image = experiment_copy.images.get_image_stack(time_point, self._measurement_channel)
            if intensity_image is None:
                continue
            segmentation_image = experiment_copy.images.get_image_stack(time_point, self._segmentation_channel)
            if segmentation_image is None:
                continue

            if intensity_image.shape != segmentation_image.shape:
                raise UserError("Image shape mismatch", f"Intensity image and segmentation image have "
                                f"different shapes at time point {time_point.time_point_number()} for experiment {experiment_copy.name}.")

            # Calculate the background intensity using the segmentation image
            background_mask = segmentation_image == 0  # Assuming background is labeled as 0
            background_intensity = intensity_image[background_mask]
            intensity_sum += background_intensity.sum()
            intensity_area += background_mask.sum()

        if intensity_area == 0:
            raise UserError("No background pixels found", f"No background pixels were found in the"
                            f" segmentation images for experiment {experiment_copy.name}."
                            f" Apparently, the segmentation images do not contain any background pixels (labeled as 0).")
        return float(intensity_sum / intensity_area)

    def use_data(self, tab: SingleGuiTab, data: Any):
        intensity_calculator.set_intensity_background(tab.experiment, data, intensity_key=self._intensity_to_correct)
        tab.undo_redo.mark_unsaved_changes()

    def on_finished(self, data: Iterable[Any]):
        dialog.popup_message("Background recorded", f"Background recorded for intensity key '{self._intensity_to_correct}'.")

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(images=True, name=True)


class _BackgroundSubtractionVisualizer(ExitableImageVisualizer):
    """Used to record the background for a given intensity key using pre-existing segmentation images.

    Only a single background value is recorded for each experiment, which is the average intensity of all pixels
    outside the segmented objects across all time points. In the Parameters menu, you must select which intensity
    to correct, which channel to use for the intensity measurement, and which channel to use for the segmentation.
    Measurement channel should match the channel that was originally used to measure that intensity."""

    _measurement_channel: ImageChannel | None = None
    _segmentation_channel: ImageChannel | None = None
    _intensity_to_correct: str | None = None

    def get_extra_menu_options(self) -> dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Parameters//Channels-Set measurement channel...": self._set_measurement_channel,
            "Parameters//Channels-Set segmentation channel...": self._set_segmentation_channel,
            "Parameters//Intensity-Set intensity to correct...": self._set_intensity_to_correct,
            "Edit//Background-Record background": self._record_background,
        }

    def _get_figure_title(self) -> str:
        return (f"Background correction (pre-existing segmentation)\n"
                f"Time point {self._time_point.time_point_number()}    (z={self._get_figure_title_z_str()}, "
                f"c={self._get_figure_title_channel_str()})")

    def _find_available_channels(self) -> set[ImageChannel]:
        """Finds all channels that are available in all open experiments."""
        channels = set()
        for experiment in self._window.get_active_experiments():
            for channel in experiment.images.get_channels():
                channels.add(channel)
        return channels

    def _set_measurement_channel(self):
        """Prompts the user for a new value of self._measurement_channel."""
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

    def _set_segmentation_channel(self):
        """Prompts the user for a new value of self._segmentation_channel."""
        current_channel = self._window.display_settings.image_channel
        if self._segmentation_channel is not None:
            current_channel = self._segmentation_channel
        channel_count = len(self._find_available_channels())

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            self._segmentation_channel = ImageChannel(index_zero=new_channel_index - 1)
            self.refresh_data()

    def _set_intensity_to_correct(self):
        """Prompts the user for an intensity key to correct."""
        intensity_keys = _find_intensity_keys(self._window)
        if len(intensity_keys) == 0:
            raise UserError("No intensity keys found", "No intensity keys were found. Please record some intensities first.")

        key_index = option_choose_dialog.prompt_list("Intensity key", "Which intensity key do you want to correct?", "Intensity", intensity_keys)
        if key_index is not None:
            self._intensity_to_correct = intensity_keys[key_index]
            self.update_status(f"Selected intensity key to correct: \"{self._intensity_to_correct}\".")

    def _record_background(self):
        """Records the background for the selected intensity key using the selected segmentation channel."""
        if self._measurement_channel is None:
            raise UserError("Measurement channel not set", "Please set a measurement channel first using "
                             "the Parameters menu. This must be the same channel that was used to record the intensity key you want to correct.")
        if self._segmentation_channel is None:
            raise UserError("Segmentation channel not set", "Please set a segmentation channel first."
                            " Any pixel outside of the segmented objects will be used to define the background.")
        if self._intensity_to_correct is None:
            intensity_keys = _find_intensity_keys(self._window)
            if len(intensity_keys) == 1:
                # No need to select an intensity, there is only one available.
                self._intensity_to_correct = intensity_keys[0]
            else:
                raise UserError("Intensity key not set", "Please set an intensity key to correct first.")

        worker_job.submit_job(self._window, _RecordBackgroundJob(
                              measurement_channel=self._measurement_channel,
                              segmentation_channel=self._segmentation_channel,
                              intensity_to_correct=self._intensity_to_correct))
        self.update_status(f"Recording background for intensity with key \"{self._intensity_to_correct}\"...")
