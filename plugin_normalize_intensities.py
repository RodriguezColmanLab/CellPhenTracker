"""Used to normalize intensities in time, Z or just overall."""
from enum import Enum, auto
from typing import Any, Optional

from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.text_popup.text_popup import RichTextPopup


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Intensity//Record-Normalize intensities//Normalize intensities...":
            lambda: _normalize_intensities(window),
        "Intensity//Record-Normalize intensities//Remove normalization...":
            lambda: _remove_normalization(window)
    }


class _NormalizationMethod(Enum):
    OVERALL = auto()
    TIME = auto()
    Z = auto()


class NormalizationPopup(RichTextPopup):

    _window: Window

    _intensity_key: str
    _intensity_key_source: str
    _normalization_method: _NormalizationMethod

    def __init__(self, window: Window):
        self._window = window

        intensity_keys = sorted(_get_all_intensity_keys(window))
        if len(intensity_keys) == 0:
            intensity_keys = [intensity_calculator.DEFAULT_INTENSITY_KEY]  # Default intensity key, if none were measured yet
        self._intensity_key = intensity_keys[0]
        self._intensity_key_source = intensity_keys[0]
        self._normalization_method = _NormalizationMethod.OVERALL

    def get_title(self) -> str:
        """Returns the title for the whole website."""
        return "Intensity normalization"

    def navigate(self, url: str) -> Optional[str]:
        if url.startswith("change_intensity_key/"):
            self._intensity_key = url[len("change_intensity_key/"):]
            return self._main_page()
        if url.startswith("change_intensity_source/"):
            self._intensity_key_source = url[len("change_intensity_source/"):]
            return self._main_page()
        if url.startswith("change_normalization_method/"):
            method_name = url[len("change_normalization_method/"):]
            self._normalization_method = _NormalizationMethod[method_name]
            return self._main_page()
        if url == "apply":
            return self._apply_normalization()
        if url == RichTextPopup.INDEX:
            return self._main_page()
        raise UserError("Unhandled URL", "Don't know how to open " + url)

    def _main_page(self) -> str:
        """Returns the Markdown for the main page."""
        return f"""
# Intensity normalization
This plugin allows you to normalize the measured intensities. Please be careful, as you may obscure your actual measurement.

**Intensity to normalize:** `{self._intensity_key}`  
{self._link_to_alternative_keys()}

**Normalize using:** `{self._intensity_key_source}`  
{self._link_to_alternative_intensity_sources()}

**Normalization method:** {self._normalization_method.name.lower()}  
{self._link_to_alternative_normalization_methods()}

[Apply](apply)

## More information
There are three normalization methods available:

* **Overall normalization**: The plugin will multiply the intensities so that the overall median intensity is 1. This is the default normalization method.
* **Time normalization**: The plugin will do an exponential fit, and then multiply the intensities so that the fitted intensity is 1 at every time point. This can be used to correct for photobleaching.
* **Z normalization**: The plugin will do an exponential fit, and then multiply the intensities so that the fitted intensity is 1 at every z-position. This can be used to correct for light scattering.

For the time and z normalization methods, it's important to normalize a reporter signal using another intensity that is not expected to change over time or z, like a H2B reporter or DAPI. Otherwise,
you might obscure the actual changes in the reporter signal.

You can use the "Intensity" > "Verify intensities" menu to check the intensities before and after normalization, to see if the normalization worked as expected.
        """

    def _link_to_alternative_keys(self) -> str:
        """Returns links to use the alternative intensity keys, if there are any."""
        all_intensity_keys = _get_all_intensity_keys(self._window)
        if self._intensity_key in all_intensity_keys:
            all_intensity_keys.remove(self._intensity_key)
        if len(all_intensity_keys) == 0:
            return ""
        result = "Change to: "
        for intensity_key in all_intensity_keys:
            result += f"[{intensity_key}](change_intensity_key/{intensity_key}) "
        return result

    def _link_to_alternative_intensity_sources(self) -> str:
        """Returns links to use the alternative intensity sources."""
        all_intensity_keys = _get_all_intensity_keys(self._window)
        if self._intensity_key_source in all_intensity_keys:
            all_intensity_keys.remove(self._intensity_key_source)
        if len(all_intensity_keys) == 0:
            return ""
        result = "Change to: "
        for intensity_key in all_intensity_keys:
            result += f"[{intensity_key}](change_intensity_source/{intensity_key}) "
        return result

    def _link_to_alternative_normalization_methods(self) -> str:
        """Returns links to use the alternative normalization methods."""
        result = ""
        for source in _NormalizationMethod:
            if source != self._normalization_method:
                result += f"[Change to '{source.name.lower()}'](change_normalization_method/{source.name}) "
        return result

    def _apply_normalization(self):
        time_correction = self._normalization_method == _NormalizationMethod.TIME
        z_correction = self._normalization_method == _NormalizationMethod.Z

        success_count = 0
        failure_count = 0
        messages = []
        for tab in self._window.get_gui_experiment().get_active_tabs():
            experiment = tab.experiment
            result = intensity_calculator.perform_intensity_normalization(experiment, background_correction=False,
                       z_correction=z_correction, time_correction=time_correction,
                       intensity_key=self._intensity_key,
                       intensity_key_source=self._intensity_key_source)
            if result.normalized:
                success_count += 1
            else:
                failure_count += 1
            if result.message is not None:
                messages.append(f"Experiment '{experiment.name}': {result.message}")
            tab.undo_redo.mark_unsaved_changes()

        if success_count > 0 and failure_count > 0:
            overall_message = f"Normalization applied successfully to {success_count} experiments, but failed for {failure_count} experiments."
        elif success_count + failure_count == 1:
            if success_count == 1:
                overall_message = f"Normalization applied successfully."
            else:
                overall_message = f"Normalization failed."
        elif success_count > 0:
            overall_message = f"Normalization applied successfully to {success_count} experiments."
        else:
            overall_message = f"Normalization failed for all {failure_count} experiments."

        if len(messages) > 0:
            overall_message += "\n\nDetails:\n\n"
            for message in messages:
                overall_message += "* " + message + "\n"

        overall_message += (f"\n\nPlease note that any existing normalizations for the intensity '{self._intensity_key}' have been replaced."
                            f"\n\nPlease check the intensities using the 'Intensity' > 'Verify intensities' menu to see if the normalization worked as expected.")
        return overall_message


def _get_all_intensity_keys(window: Window) -> set[str]:
    """Gets all intensity keys available for all experiments. Only considers regular intensity keys, as we can only
    normalize those."""
    keys = set()
    for experiment in window.get_active_experiments():
        keys |= set(intensity_calculator.get_regular_intensity_keys(experiment))
    return keys


def _verify_saved_intensities(window: Window):
    """Raises a UserError if no intensities were stored."""
    if len(_get_all_intensity_keys(window)) == 0:
        raise UserError("No intensities", "No intensities were measured. Please do so first.")


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


def _normalize_intensities(window: Window):
    _verify_saved_intensities(window)
    dialog.popup_rich_text(NormalizationPopup(window))


def _remove_normalization(window: Window):
    _verify_saved_intensities(window)
    if not dialog.popup_message_cancellable("Normalization", "The normalization will be removed, so that"
                                                             " only a background correction (if any) will remain."):
        return

    intensity_keys = _prompt_intensity_keys(window)
    for tab in window.get_gui_experiment().get_all_tabs():
        experiment = tab.experiment
        for intensity_key in intensity_keys:
            intensity_calculator.remove_intensity_normalization(experiment, intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()

