import cellpose.models
import math
import numpy
import os
import skimage
import tifffile
from numpy import ndarray
from tqdm import tqdm

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelAppendingImageLoader
from organoid_tracker.image_loading.folder_image_loader import FolderImageLoader
from organoid_tracker.imaging import list_io
from . import _configuration

_TARGET_RESOLUTION_UM = (1, 1, 1)  # Z, Y, X


def _find_time_points_without_masks(experiment: Experiment, output_folder: str) -> list[TimePoint]:
    """Finds all time points in the experiment that do not yet have a mask file in the output folder."""
    time_points = list()
    for time_point in experiment.images.time_points():
        output_file = os.path.join(output_folder, f"masks_t{time_point.time_point_number():04d}.tif")
        if not os.path.exists(output_file):
            time_points.append(time_point)
    return time_points


def main():
    config = _configuration.SegmentationConfig()
    config_file = ConfigFile("segment_cells_3d_shapy_blobs")
    config.read_config(config_file)
    config_file.save_and_exit_if_changed()

    image_channel = ImageChannel(index_one=config.image_channel)
    model = cellpose.models.CellposeModel(gpu=True)

    all_experiments_list_file = os.path.join(config.output_folder, "All experiments" + list_io.FILES_LIST_EXTENSION)
    if os.path.exists(all_experiments_list_file):
        os.unlink(all_experiments_list_file)  # Delete existing file, otherwise we would append to it

    for i, experiment in enumerate(list_io.load_experiment_list_file(config.input_dataset_file)):
        print(f"\nSegmenting experiment: {experiment.name}")

        output_folder = os.path.join(config.output_folder, f"{i + 1}. {experiment.name.get_save_name()}")
        os.makedirs(output_folder, exist_ok=True)

        remaining_time_points = _find_time_points_without_masks(experiment, output_folder)

        for time_point in tqdm(remaining_time_points):
            # Load the image as a numpy array
            image = experiment.images.get_image_stack(time_point, image_channel)

            # Find how we need to rescale the image to get to the target resolution
            original_size = image.shape
            resolution = experiment.images.resolution()
            z_rescale_factor = resolution.pixel_size_zyx_um[0] / _TARGET_RESOLUTION_UM[0]
            y_rescale_factor = resolution.pixel_size_zyx_um[1] / _TARGET_RESOLUTION_UM[1]
            x_rescale_factor = resolution.pixel_size_zyx_um[2] / _TARGET_RESOLUTION_UM[2]

            # Rescale the image to the target resolution
            resized_image = skimage.transform.rescale(
                image,
                (z_rescale_factor, y_rescale_factor, x_rescale_factor),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
                channel_axis=None
            )
            del image  # Save some memory

            # Run cellpose on the image
            masks, flows, styles = model.eval(resized_image, resample=False, batch_size=8, do_3D=True,
                                              z_axis=0,
                                              normalize={"percentile": [config.min_percentile, config.max_percentile]})
            del resized_image, flows, styles  # Save some memory

            # Refine the masks if we rescaled the image
            if (z_rescale_factor, y_rescale_factor, x_rescale_factor) != (1.0, 1.0, 1.0):
                masks = skimage.transform.resize(masks, original_size, order=0, preserve_range=True, anti_aliasing=False).astype(masks.dtype)
                _refine_masks(masks, 1 / min(z_rescale_factor, y_rescale_factor, x_rescale_factor), gaussian_cutoff=config.mask_refinement_cutoff)

            # Write the result
            output_file = os.path.join(output_folder, f"masks_t{time_point.time_point_number():04d}.tif")
            tifffile.imwrite(output_file, masks, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

        # Add the newly written masks as a channel to the experiment
        masks_image_loader = FolderImageLoader(os.path.abspath(output_folder), "masks_t{time:04d}.tif",
                                               experiment.images.first_time_point_number(), experiment.images.last_time_point_number(), 1, 1)
        experiment.images.image_loader(ChannelAppendingImageLoader([experiment.images.image_loader(), masks_image_loader]))

        # Add the experiment to the all experiments list
        list_io.save_experiment_list_file([experiment], all_experiments_list_file, append_to_file=True)


def _refine_masks(masks: ndarray, enlargement_factor: float, gaussian_cutoff: float = 0.5):
    """Takes masks that have been rescaled up using a crappy nearest-neighbor method and refines them using a Gaussian
    blur, so that they have a smooth shape again."""
    if enlargement_factor < 1.0:
        return  # No need to refine

    padding = int(enlargement_factor + 3)
    new_masks = numpy.zeros((masks.shape[0] + padding * 2, masks.shape[1] + padding * 2, masks.shape[2] + padding * 2), dtype=masks.dtype)

    for mask in skimage.measure.regionprops(masks):
        # Find size of the mask of the cell
        mask_width = int(mask.bbox[5] - mask.bbox[2])
        mask_height = int(mask.bbox[4] - mask.bbox[1])
        mask_depth = int(mask.bbox[3] - mask.bbox[0])

        mask_padded_width = mask_width + padding * 2
        mask_padded_height = mask_height + padding * 2
        mask_padded_depth = mask_depth + padding * 2

        # Create a padded float image for the mask, placing the mask in the center
        mask_float_image = numpy.zeros((mask_padded_depth, mask_padded_height, mask_padded_width), dtype=numpy.float32)
        mask_float_image[padding:padding + mask_depth, padding:padding + mask_height, padding:padding + mask_width] = mask.image

        # Do a Gaussian blur on the mask to smooth the edges
        mask_float_image = skimage.filters.gaussian(mask_float_image, sigma=math.sqrt(enlargement_factor))

        # Place the mask in the new_masks array
        new_masks_crop = new_masks[mask.bbox[0]:mask.bbox[0] + mask_padded_depth, mask.bbox[1]:mask.bbox[1] + mask_padded_height, mask.bbox[2]:mask.bbox[2] + mask_padded_width]
        new_masks_crop[mask_float_image > gaussian_cutoff] = mask.label

    # Copy back the refined masks to the original mask array
    masks[:, :] = new_masks[padding:padding + masks.shape[0], padding:padding + masks.shape[1], padding:padding + masks.shape[2]]


if __name__ == "__main__":
    main()