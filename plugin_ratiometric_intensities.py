from functools import partial
from typing import Any, Callable

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.undo_redo import UndoableAction, ReversedAction
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator


class _AddRatiometricIntensityAction(UndoableAction):
    """Adds (or removes) a ratiometric intensity."""

    _ratiometric_key: str
    _intensity_key_1: str
    _intensity_key_2: str
    _undo_function: Callable[[Experiment], Any]

    def __init__(self, ratiometric_key: str, intensity_key_1: str, intensity_key_2: str):
        self._ratiometric_key = ratiometric_key
        self._intensity_key_1 = intensity_key_1
        self._intensity_key_2 = intensity_key_2
        self._undo_function = lambda e: intensity_calculator.remove_ratiometric_intensity(e, self._ratiometric_key)

    def do(self, experiment: Experiment):
        # We replace the undo function with the provided one. There might have been a previous ratiometric intensity
        # with the same name, and the provided undo function will restore that one instead of just removing the new one.
        self._undo_function = intensity_calculator.add_ratiometric_intensity(
            experiment, self._ratiometric_key, self._intensity_key_1, self._intensity_key_2)

    def undo(self, experiment: Experiment):
        self._undo_function(experiment)


def get_menu_items(window: Window) -> dict[str, Any]:

    # Add a button to add a new ratiometric intensity
    return_dict = {
        "Intensity//Record-Ratiometric intensities//New ratiometric intensity...": lambda: _add_ratiometric_intensity(window)
    }

    # Also add buttons to remove existing ratiometric intensities
    ratiometric_intensity_keys = set()
    for experiment in window.get_active_experiments():
        for intensity_key in intensity_calculator.get_ratiometric_intensity_keys(experiment):
            ratiometric_intensity_keys.add(intensity_key)
    ratiometric_intensity_keys = sorted(ratiometric_intensity_keys)
    for intensity_key in ratiometric_intensity_keys:
        return_dict["Intensity//Record-Ratiometric intensities//Remove ratiometric intensity//Intensity-" + intensity_key]\
            = partial(_remove_ratiometric_intensity, window, intensity_key)

    return return_dict


def _add_ratiometric_intensity(window: Window):
    regular_intensity_keys = set()
    for experiment in window.get_active_experiments():
        for intensity_key in intensity_calculator.get_regular_intensity_keys(experiment):
            regular_intensity_keys.add(intensity_key)
    regular_intensity_keys = sorted(regular_intensity_keys)
    if len(regular_intensity_keys) < 2:
        raise UserError("No regular intensities found", "Ratiometric intensities are ratios of two regular"
                                                        " intensity. Please record two of them first.")

    chosen_index = option_choose_dialog.prompt_list("First intensity", "Choose the first intensity (the numerator)",
                                                    "Intensity:", regular_intensity_keys)
    if chosen_index is None:
        return
    intensity_key_1 = regular_intensity_keys[chosen_index]

    remaining_intensity_keys = regular_intensity_keys
    del remaining_intensity_keys[chosen_index]

    if len(regular_intensity_keys) == 1:
        intensity_key_2 = remaining_intensity_keys[0]
        if not dialog.popup_message_cancellable("Only one intensity left", f"Only one regular intensity is left (\"{intensity_key_2}\"),"
                                                                           f" so it will be used as the second intensity (the denominator)."):
            return
    else:
        chosen_index = option_choose_dialog.prompt_list("Second intensity", "Choose the second intensity (the denominator)",
                                                        "Intensity:", remaining_intensity_keys)
        if chosen_index is None:
            return
        intensity_key_2 = remaining_intensity_keys[chosen_index]

    intensity_name = dialog.prompt_str("Ratiometric intensity name", f"Enter a name for this ratiometric intensity"
                                       f" of \"{intensity_key_1}\" over \"{intensity_key_2}\" (e.g. \"green_over_red\").")
    if intensity_name is None:
        return
    if intensity_name in regular_intensity_keys:
        raise UserError("Name already exists", f"An intensity with the name \"{intensity_name}\" already exists."
                                               f" Please choose a different name.")

    for tab in window.get_gui_experiment().get_active_tabs():
        regular_intensity_keys_of_experiment = intensity_calculator.get_regular_intensity_keys(tab.experiment)
        if intensity_key_1 not in regular_intensity_keys_of_experiment or intensity_key_2 not in regular_intensity_keys_of_experiment:
            continue  # Not present for this experiment, just for others

        action = _AddRatiometricIntensityAction(intensity_name, intensity_key_1, intensity_key_2)
        tab.undo_redo.do(action, tab.experiment)
    window.redraw_all()
    window.set_status(f"Added ratiometric intensity \"{intensity_name}\" as the ratio of \"{intensity_key_1}\" over \"{intensity_key_2}\"."
                      f" It is now available for viewing and plotting like any other intensity.")


def _remove_ratiometric_intensity(window: Window, intensity_key: str):
    if not dialog.popup_message_cancellable("Remove ratiometric intensity", f"Are you sure you want to remove"
                                            f" the ratiometric intensity \"{intensity_key}\"?"):
        return

    for tab in window.get_gui_experiment().get_active_tabs():
        old_keys = intensity_calculator.get_intensities_for_ratiometric_intensity(tab.experiment, intensity_key)
        if old_keys is None:
            continue  # Not present for this experiment, just for others

        action = ReversedAction(_AddRatiometricIntensityAction(intensity_key, old_keys[0], old_keys[1]))
        tab.undo_redo.do(action, tab.experiment)
    window.redraw_all()
    window.set_status(f"Removed ratiometric intensity \"{intensity_key}\".")
