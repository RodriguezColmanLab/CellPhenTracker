from organoid_tracker.config import ConfigFile, config_type_int, config_type_float
from organoid_tracker.imaging import list_io


def _config_type_tuple3_float(value: str) -> tuple[float, float, float]:
    parts = value.strip("()").split(",")
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated values")
    return float(parts[0]), float(parts[1]), float(parts[2])


class SegmentationConfig:
    """Configuration for the segmentation."""
    input_dataset_file: str = "Dataset" + list_io.FILES_LIST_EXTENSION
    output_folder: str = "Segmentation masks"
    target_resolution_zyx_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
    image_channel: int = 1
    min_percentile: float = 1.0
    max_percentile: float = 99.0
    mask_refinement_cutoff: float = 0.5

    def read_config(self, config_file: ConfigFile):
        self.input_dataset_file = config_file.get_or_default("input_dataset_file", self.input_dataset_file,
                                                             comment="Path to the dataset file listing the images to segment.")
        self.output_folder = config_file.get_or_default("output_folder", self.output_folder,
                                                       comment="Folder to store the segmentation masks in.")
        self.target_resolution_zyx_um = config_file.get_or_default("target_resolution_zyx_um", self.target_resolution_zyx_um,
                                                                comment="Target resolution (Z, Y, X) in microns for the segmentation. Images will be rescaled to this resolution before segmentation, and the masks will be rescaled back to the original image resolution.",
                                                                type=_config_type_tuple3_float)
        self.image_channel = config_file.get_or_default("image_channel", self.image_channel,
                                                       comment="Image channel to use for segmentation (1-based).",
                                                       type=config_type_int)
        self.min_percentile = config_file.get_or_default("min_percentile", self.min_percentile,
                                                         comment="Minimum percentile during intensity rescaling for segmentation.",
                                                         type=config_type_float)
        self.max_percentile = config_file.get_or_default("max_percentile", self.max_percentile,
                                                         comment="Maximum percentile during intensity rescaling for segmentation.",
                                                         type=config_type_float)
        self.mask_refinement_cutoff = config_file.get_or_default("mask_refinement_cutoff", self.mask_refinement_cutoff,
                                                                comment="After segmentation, the masks are blown up so that they match"
                                                                        " the original image resolution. A Gaussian blur is applied to the blown-up masks, "
                                                                        " and all pixels with a value above this cutoff are included in the final mask."
                                                                        " If your masks are too small, decrease this value; if they are too large, increase it.",
                                                                type=config_type_float)