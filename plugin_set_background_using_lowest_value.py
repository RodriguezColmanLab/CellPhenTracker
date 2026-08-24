"""If the organoid has a membrane marker, then that can be used for segmentation."""
from typing import Any

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Intensity//Record-Background correction//Basic-Correct using lowest measured value...":
            lambda: _correct_background(window),
        "Intensity//Record-Background correction//Remove-Remove background correction...":
            lambda: _remove_background_correction(window)
    }


def _get_all_intensity_keys(window: Window) -> set[str]:
    """Gets all intensity keys available for all experiments. Only considers regular intensity keys, as we can only
    normalize those."""
    keys = set()
    for experiment in window.get_active_experiments():
        keys |= set(intensity_calculator.get_regular_intensity_keys(experiment))
    return keys


def _prompt_intensity_keys(window: Window) -> list[str]:
    """If there are more than one intensity keys, this prompts the user which ones should be used. Returns an empty list
     if the user pressed Cancel, or if there were no intensities selected."""
    intensity_keys = list(_get_all_intensity_keys(window))

    if len(intensity_keys) > 1:
        intensity_key_indices = option_choose_dialog.prompt_list_multiple("Intensities", "We found multiple intensities. Which"
                                                                   " ones should we normalize? Select all that apply",
                                                                   "Intensity keys:", intensity_keys)
        if intensity_key_indices is None:
            return []  # Cancelled
        if len(intensity_key_indices) == 0:
            # User pressed OK, but didn't select anything. Likely in error, so notify the user.
            raise UserError("No keys selected", "No intensity keys were selected. Please check the boxes of the"
                                                " intensities that you want to normalize.")
        return [intensity_keys[i] for i in intensity_key_indices]
    return intensity_keys


def _find_background_per_pixel(experiment: Experiment, intensity_key: str) -> float:
    """Finds the lowest intensity value in the experiment for the given intensity key, and returns that value divided by
    the number of pixels in the cell. This is used to set the background for the intensity key."""
    min_intensity_per_px = None

    positions = experiment.positions
    for position, intensity in positions.find_all_positions_with_data(intensity_key):
        volume = positions.get_position_data(position, intensity_key + "_volume")
        if volume is None or volume <= 0:
            continue
        intensity_per_px = intensity / volume
        if min_intensity_per_px is None or intensity_per_px < min_intensity_per_px:
            min_intensity_per_px = intensity_per_px

    return min_intensity_per_px


def _correct_background(window: Window):
    if len(_get_all_intensity_keys(window)) == 0:
        raise UserError("No intensities", "No intensities were measured. Please do so first.")
    if not dialog.popup_message_cancellable("Normalization", "The normalization of the intensities will be changed. "
                                            "The lowest found intensity in the experiment is used for setting the "
                                            "background. In addition, the intensities will be multiplied to obtain "
                                            "a median intensity of 1 at each z position."):
        return

    intensity_keys = _prompt_intensity_keys(window)
    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        for intensity_key in intensity_keys:
            background_per_px = _find_background_per_pixel(experiment, intensity_key)
            intensity_calculator.set_intensity_background(experiment, background_per_px, intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()


def _remove_background_correction(window: Window):
    if len(_get_all_intensity_keys(window)) == 0:
        raise UserError("No intensities", "No intensities were measured. Please do so first.")
    if not dialog.popup_message_cancellable("Remove Background Correction", "This will remove the"
              " background correction (as well as any normalization) for the selected intensities."):
        return

    intensity_keys = _prompt_intensity_keys(window)
    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        for intensity_key in intensity_keys:
            intensity_calculator.set_intensity_background(experiment, None, intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()

