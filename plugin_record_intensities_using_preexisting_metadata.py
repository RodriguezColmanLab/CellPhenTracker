"""Uses a simple sphere of a given radius for segmentation"""
from typing import Dict, Any

from organoid_tracker.core import UserError
from organoid_tracker.gui import option_choose_dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Record intensities//Record-Record using pre-existing metadata...": lambda: _record_intensities(window)
    }

def _record_intensities(window: Window):
    # Gather possible metadata and existing keys
    existing_keys = set()
    possible_keys = set()
    for experiment in window.get_active_experiments():
        existing_keys |= set(intensity_calculator.get_intensity_keys(experiment))

        # Find possible keys of intensities
        names_and_types = experiment.positions.get_data_names_and_types()
        for data_name, data_type in names_and_types.items():
            if data_type != float:
                continue  # Skip non-numeric metadata

            if data_name.endswith("_volume") and data_name[:-len("_volume")] in names_and_types:
                continue  # Skip volume of an intensity

            possible_keys.add(data_name)

    # Remove existing keys from intensities
    for existing_key in existing_keys:
        if existing_key in possible_keys:
            possible_keys.remove(existing_key)

    # Raise error if nothing was found
    if len(possible_keys) == 0:
        raise UserError("No metadata found", "No suitable metadata found. Any numeric position metadata is fine."
                                             " You can import metadata through the File menu.")

    possible_keys = list(possible_keys)
    result = option_choose_dialog.prompt_list_multiple("Intensity keys", "Which metadata would you like to use as an intensity?"
          " Note: for normalization purposes, it is assumed that the intensity was measured using the same volume for all cells.", "Metadata keys", possible_keys)
    if result is None:
        return
    if len(result) == 0:
        raise UserError("No options selected", "No options were selected. Cannot do anything.")

    new_intensity_keys = [possible_keys[i] for i in result]
    for experiment in window.get_active_experiments():
        for new_intensity_key in new_intensity_keys:
            raw_intensities = dict(experiment.positions.find_all_positions_with_data(new_intensity_key))
            volumes = dict(((position, 1) for position in raw_intensities.keys()))
            intensity_calculator.set_raw_intensities(experiment, raw_intensities, volumes, intensity_key=new_intensity_key)