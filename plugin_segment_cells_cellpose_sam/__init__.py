from typing import Callable, Any

from organoid_tracker.gui.window import Window


def get_commands() -> dict[str, Callable[[list[str]], int]]:
    # Command-line commands, used in python organoid_tracker.py <command>
    return {
        "segment_cells_3d_cellpose_sam": _segment_cells
    }

def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "Tools//Segment-Segment with scaled Cellpose-SAM model...": lambda: _create_segmentation_script(window),
    }

def _segment_cells(args: list[str]) -> int:
    print("Hi! This is the Cellpose-SAM 3D cell segmentation plugin.")
    from . import _cellpose_segmentation
    _cellpose_segmentation.main()
    return 0

def _create_segmentation_script(window: Window):
    from . import _script_generator
    _script_generator.create_segmentation_script(window)
