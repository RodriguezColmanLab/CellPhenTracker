"""Plugin to remove existing regular intensities from the experiments."""
from functools import partial
from typing import Any

from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator


def get_menu_items(window: Window) -> dict[str, Any]:

    return_dict = dict()

    # Add buttons to remove existing regular intensities
    regular_intensity_keys = set()
    for experiment in window.get_active_experiments():
        for intensity_key in intensity_calculator.get_regular_intensity_keys(experiment):
            regular_intensity_keys.add(intensity_key)
    regular_intensity_keys = sorted(regular_intensity_keys)
    for intensity_key in regular_intensity_keys:
        return_dict["Intensity//Record-Record intensities//Remove-Remove intensities//" + intensity_key]\
            = partial(_remove_intensity, window, intensity_key)

    return return_dict


def _remove_intensity(window: Window, intensity_key: str):
    if not dialog.prompt_yes_no(f"Removing '{intensity_key}'", f"Are you sure you want to remove the"
                                f" intensity '{intensity_key}'? This cannot be undone; you would need to measure the intensity again."):
        return
    for tab in window.get_gui_experiment().get_active_tabs():
        intensity_calculator.remove_intensities(tab.experiment, intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()

    window.redraw_all()  # To update the menu
    if len(list(window.get_active_experiments())) > 1:
        window.set_status(f"Removed intensity '{intensity_key}' from all experiments.")
    else:
        window.set_status(f"Removed intensity '{intensity_key}' from the experiment.")