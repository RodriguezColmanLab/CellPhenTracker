"""If the organoid has a membrane marker, then that can be used for segmentation."""
from typing import Dict, Any

from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Normalize intensities//Normalize with background and z correction...":
            lambda: _normalize_with_background_and_z(window),
        "Intensity//Record-Normalize intensities//Normalize with background correction...":
            lambda: _normalize_with_background(window),
        "Intensity//Record-Normalize intensities//Normalize without corrections...":
            lambda: _normalize_without_background(window),
        "Intensity//Record-Normalize intensities//Remove normalization...":
            lambda: _remove_normalization(window)
    }


def _normalize_with_background_and_z(window: Window):
    if not dialog.popup_message_cancellable("Normalization", "The normalization of the intensities will be changed.\n"
                                            "The lowest found intensity in the experiment is used for setting the\n"
                                            "background. In addition, the intensities have been multiplied to obtain\n"
                                            "a median intensity of 1 at each z position."):
        return

    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        intensity_calculator.perform_intensity_normalization(experiment, background_correction=True,
                                                                                                z_correction=True)
        tab.undo_redo.mark_unsaved_changes()


def _normalize_with_background(window: Window):
    if not dialog.popup_message_cancellable("Normalization", "The normalization of the intensities will be changed.\n"
                                            "The lowest found intensity in the experiment is used for setting the\n"
                                            "background. In addition, the intensities have been multiplied to obtain\n"
                                            "an overall median intensity of 1."):
        return

    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        intensity_calculator.perform_intensity_normalization(experiment, background_correction=True,
                                                                                                z_correction=False)
        tab.undo_redo.mark_unsaved_changes()


def _normalize_without_background(window: Window):
    if not dialog.popup_message_cancellable("Normalization", "All intensities will be normalized. No background\n"
                                            "correction is used. Still, the intensities have been multiplied to\n"
                                            "obtain an overall median intensity of 1."):
        return

    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        intensity_calculator.perform_intensity_normalization(experiment, background_correction=False, z_correction=False)
        tab.undo_redo.mark_unsaved_changes()

def _remove_normalization(window: Window):
    if not dialog.popup_message_cancellable("Normalization", "The normalization will be removed, so that script will\n"
                                                             "use the raw values again."):
        return

    for tab in window.get_gui_experiment().get_all_tabs():
        experiment = tab.experiment
        intensity_calculator.remove_intensity_normalization(experiment)
        tab.undo_redo.mark_unsaved_changes()

