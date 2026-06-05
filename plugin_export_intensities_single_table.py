import csv
from enum import Enum, auto
from typing import Any

from organoid_tracker.core import UserError, min_none, max_none
from organoid_tracker.gui import option_choose_dialog, dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator


class _ExportVariant(Enum):
    RAW_SUMMED = auto()
    NORMALIZED_SUMMED = auto()
    NORMALIZED_PER_PIXEL = auto()

    def is_normalized(self) -> bool:
        return self in (_ExportVariant.NORMALIZED_SUMMED, _ExportVariant.NORMALIZED_PER_PIXEL)

    def is_per_pixel(self) -> bool:
        return self == _ExportVariant.NORMALIZED_PER_PIXEL

    def display_name(self) -> str:
        if self == _ExportVariant.NORMALIZED_PER_PIXEL:
            return "Per-pixel, after applying any normalization"
        if self == _ExportVariant.RAW_SUMMED:
            return "Raw (summed, skipping any normalization)"
        if self == _ExportVariant.NORMALIZED_SUMMED:
            return "Summed, after applying any normalization"
        raise ValueError("Unknown export variant: " + str(self))


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "File//Export-Export intensities//CSV, tracks as columns, single intensity...": lambda: _export_intensities(window)
    }


def _export_intensities(window: Window):
    intensity_keys = set()
    for experiment in window.get_active_experiments():
        if not experiment.links.has_links():
            raise UserError("No links available", f"The experiment '{experiment.name}' has no cell"
                            f" tracking available. Please track the cells in this experiment before using this export option.")
        intensity_keys |= set(intensity_calculator.get_intensity_keys(experiment))
    if len(intensity_keys) == 0:
        raise UserError("No recorded intensities", "No intensities were recorded. Please record some using the Intensity menu.")

    # Prompt for intensity key
    intensity_keys = list(intensity_keys)
    intensity_index = option_choose_dialog.prompt_list("Intensity key", "Which intensity should we export?", "Intensity key:", intensity_keys)
    if intensity_index is None:
        return
    intensity_key = intensity_keys[intensity_index]

    # Prompt for intensity variant
    intensity_exports = [variant for variant in _ExportVariant]
    intensity_exports_names = [variant.display_name() for variant in _ExportVariant]
    intensity_export_index = option_choose_dialog.prompt_list("Export variant", "Which variant of the intensities should we export?", "Export variant:", intensity_exports_names)
    if intensity_export_index is None:
        return
    intensity_export = intensity_exports[intensity_export_index]

    # Prompt for export file
    export_file = dialog.prompt_save_file("Export intensities",  [("CSV file", "*.csv"), ("TSV file", "*.tsv")])
    if export_file is None:
        return

    # Export!
    _write_csv(window, export_file, intensity_export, intensity_key)


def _write_csv(window: Window, export_file: str, intensity_export: _ExportVariant, intensity_key: str):
    columns = list()
    min_time_point_number = None
    max_time_point_number = None
    for experiment in window.get_active_experiments():
        min_time_point_number = min_none(experiment.positions.first_time_point_number(), min_time_point_number)
        max_time_point_number = max_none(experiment.positions.last_time_point_number(), max_time_point_number)
    if min_time_point_number is None:
        min_time_point_number = 0
    if max_time_point_number is None:
        max_time_point_number = min_time_point_number

    columns.append(["Time point"] + list(range(min_time_point_number, max_time_point_number + 1)))
    time_point_count = max_time_point_number - min_time_point_number + 1
    for experiment in window.get_active_experiments():
        experiment_positions = experiment.positions
        for track in experiment.links.find_all_tracks():
            start = track.find_first_position()
            column = [f"{experiment.name} x{start.x:.2f} y{start.y:.2f} z{start.z:.2f} t{start.time_point_number()}"]
            column += [float("nan")] * time_point_count

            for position in track.positions():
                if intensity_export.is_normalized():
                    intensity = intensity_calculator.get_normalized_intensity(experiment, position,
                                                                              intensity_key=intensity_key,
                                                                              per_pixel=intensity_export.is_per_pixel())
                else:
                    intensity = intensity_calculator.get_raw_intensity(experiment_positions, position,
                                                                       intensity_key=intensity_key)
                if intensity is None:
                    continue
                index = position.time_point_number() - min_time_point_number + 1
                column[index] = intensity
            columns.append(column)

    delimiter = ","
    if export_file.lower().endswith(".tsv"):
        delimiter = "\t"
    with open(export_file, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        for row_index in range(len(columns[0])):
            row = []
            for column_index in range(len(columns)):
                value = ""
                if row_index < len(columns[column_index]):
                    value = columns[column_index][row_index]
                row.append(value)
            writer.writerow(row)


