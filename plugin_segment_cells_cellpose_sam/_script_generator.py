import json

import os

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import list_io
from organoid_tracker.util.run_script_creator import create_run_script


def create_segmentation_script(window: Window):
    try:
        # Check if Cellpose-SAM is installed
        import cellpose
    except ImportError:
        raise UserError("Cellpose-SAM not found",
                        "The Cellpose-SAM package is not installed. Please install it first using"
                        " `conda install conda-forge::cellpose`, then restart the program.")
    try:
        # Check if Cellpose-SAM is installed
        import cellpose.vit_sam
    except ImportError:
        raise UserError("Cellpose <=3 found instead of CellPose-SAM",
                        "It appears that Cellpose version <=3 is installed instead of Cellpose-SAM. Please"
                        " update.")

    # Ask for output folder
    output_folder = dialog.prompt_save_file("Output folder", [("*", "Folder")])
    if output_folder is None:
        return
    os.makedirs(output_folder)

    # Save dataset information
    data_structure = action.to_experiment_list_file_structure(window.get_gui_experiment().get_active_tabs())
    with open(os.path.join(output_folder, "Dataset" + list_io.FILES_LIST_EXTENSION), "w") as handle:
        json.dump(data_structure, handle)

    # Save run script
    create_run_script(output_folder, "segment_cells_3d_shapy_blobs")

    # Save config file
    current_image_channel = window.display_settings.image_channel.index_one
    config_file = ConfigFile("segment_cells_3d_shapy_blobs", folder_name=output_folder)
    from . import  _configuration
    config = _configuration.SegmentationConfig()
    config.read_config(config_file)
    config_file.set("image_channel", str(current_image_channel),
                    comment="The image channel to use for cell segmentation. This is a 1-based index.")
    config_file.save()

    # Done!
    if dialog.prompt_options("Run folder created", f"The configuration files were created successfully. Please"
                                                   f" run the types_train script from that directory:\n\n{output_folder}",
                             option_default=DefaultOption.OK, option_1="Open that directory") == 1:
        dialog.open_file(output_folder)