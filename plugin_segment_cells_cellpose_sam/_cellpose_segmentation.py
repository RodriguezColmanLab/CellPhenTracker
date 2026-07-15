import math
import os
from typing import Any

import cellpose.models
import numpy
import skimage
import tifffile
from cellpose.models import CellposeModel
from numpy import ndarray, dtype, float64
from tqdm import tqdm

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelAppendingImageLoader
from organoid_tracker.image_loading.folder_image_loader import FolderImageLoader
from organoid_tracker.imaging import list_io
from . import _configuration
from ._configuration import SegmentationConfig


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
    config_file = ConfigFile("segment_cells_3d_cellpose_sam")
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
            if image is None:
                continue

            masks = _segment_image(experiment, model, config, image)

            # Write the result
            output_file = os.path.join(output_folder, f"masks_t{time_point.time_point_number():04d}.tif")
            tifffile.imwrite(output_file, masks, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

        # Add the newly written masks as a channel to the experiment
        masks_image_loader = FolderImageLoader(os.path.abspath(output_folder), "masks_t{time:04d}.tif",
                                               experiment.images.first_time_point_number(), experiment.images.last_time_point_number(), 1, 1)
        experiment.images.image_loader(ChannelAppendingImageLoader([experiment.images.image_loader(), masks_image_loader]))

        # Add the experiment to the all experiments list
        list_io.save_experiment_list_file([experiment], all_experiments_list_file, append_to_file=True)


def _is_worth_rescaling(z_rescale_factor: float, y_rescale_factor: float, x_rescale_factor: float) -> bool:
    if z_rescale_factor < 0.98 or z_rescale_factor > 1.02:
        return True
    if y_rescale_factor < 0.98 or y_rescale_factor > 1.02:
        return True
    if x_rescale_factor < 0.98 or x_rescale_factor > 1.02:
        return True
    return False


def _segment_image(experiment: Experiment, model: CellposeModel, config: SegmentationConfig, image: ndarray) -> ndarray:
    # Find how we need to rescale the image to get to the target resolution
    resolution = experiment.images.resolution()
    rescale_factor_zyx = tuple(resolution.pixel_size_zyx_um[i] / config.target_resolution_zyx_um[i] for i in range(3))

    if config.block_size_cellpose_px <= 0:
        # Just resize at once
        return _segment_nontiled(config, model, image, rescale_factor_zyx)
    else:
        # Big image, need to handle it piece by piece
        return _segment_tiled(config, model, image, rescale_factor_zyx)


def _segment_nontiled(config: SegmentationConfig, model: CellposeModel, image: ndarray,
                      rescale_factor_zyx: tuple[float, float, float]) -> Any:
    original_size_zyx = image.shape
    if _is_worth_rescaling(*rescale_factor_zyx):
        resized_image = skimage.transform.rescale(
            image,
            rescale_factor_zyx,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
            channel_axis=None
        )
    else:
        resized_image = image
    del image  # Save some memory

    # Run cellpose on the image
    masks, flows, styles = model.eval(resized_image, resample=False, batch_size=8, do_3D=True,
                                      z_axis=0,
                                      normalize={"percentile": [config.min_percentile, config.max_percentile]})
    del resized_image, flows, styles  # Save some memory

    # Refine the masks if we rescaled the image
    if _is_worth_rescaling(*rescale_factor_zyx):
        masks = skimage.transform.resize(masks, original_size_zyx, order=0, preserve_range=True,
                                         anti_aliasing=False).astype(masks.dtype)
        enlargement_factor = 1 / min(rescale_factor_zyx) * config.mask_smoothing_factor
        _refine_masks(masks, enlargement_factor, gaussian_cutoff=config.mask_refinement_cutoff)

    return masks


def _segment_tiled(config: SegmentationConfig, model: CellposeModel, image: ndarray,
                   rescale_factor_zyx: tuple[float, float, float]) -> ndarray:
    # (Cellpose also runs in tiles, but it has a few steps were it copies the entire array. So running it on the whole
    # array at once doesn't actually work for huge images (~2000 x 2000 x 50 px on my PC).

    vmin = numpy.percentile(image, config.min_percentile)
    vmax = numpy.percentile(image, config.max_percentile)
    original_size_zyx = image.shape

    # We're going to run CellPose on blocks
    block_size_cellpose = config.block_size_cellpose_px
    overlap_cellpose = config.block_size_cellpose_overlap_px
    block_size_original = (int(block_size_cellpose / rescale_factor_zyx[0]),
                           int(block_size_cellpose / rescale_factor_zyx[1]),
                           int(block_size_cellpose / rescale_factor_zyx[2]))
    overlap_original = (int(overlap_cellpose / rescale_factor_zyx[0]),
                        int(overlap_cellpose / rescale_factor_zyx[1]),
                        int(overlap_cellpose / rescale_factor_zyx[2]))

    masks_output = numpy.zeros(original_size_zyx, dtype=numpy.uint16)
    masks_count = 0
    for z_start in range(0, image.shape[0], block_size_original[0] - overlap_original[0]):
        for y_start in range(0, image.shape[1], block_size_original[1] - overlap_original[1]):
            for x_start in range(0, image.shape[2], block_size_original[2] - overlap_original[2]):
                print(x_start, y_start, z_start)
                input_block = image[
                    z_start:z_start + block_size_original[0], y_start:y_start + block_size_original[
                        1], x_start:x_start + block_size_original[2]]
                input_block_size = input_block.shape
                if _is_worth_rescaling(*rescale_factor_zyx):
                    input_block_resized = skimage.transform.rescale(
                        input_block,
                        rescale_factor_zyx,
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                        channel_axis=None
                    )
                else:
                    input_block_resized = input_block.astype(numpy.float32)
                del input_block

                # We normalize ourselves based on the overall vmin and vmax
                input_block_resized -= vmin
                input_block_resized *= 1.0 / (vmax - vmin)
                numpy.clip(input_block_resized, 0, 1, out=input_block_resized)

                block_max = input_block_resized.max()
                if block_max < 0.33:
                    continue  # Don't bother to run CellPose, likely no nuclei here

                # Run cellpose on the image
                masks_block, flows, styles = model.eval(input_block_resized, resample=False, batch_size=8, do_3D=True,
                                                        z_axis=0,
                                                        normalize={"lowhigh": [0.0, 1.0]})
                del input_block_resized, flows, styles  # Save some memory

                # Resize the masks back to the original size
                if _is_worth_rescaling(*rescale_factor_zyx):
                    masks_block = skimage.transform.resize(masks_block, input_block_size,
                                                           order=0, preserve_range=True, anti_aliasing=False).astype(
                        masks_block.dtype)
                    enlargement_factor = 1 / min(rescale_factor_zyx) * config.mask_smoothing_factor
                    _refine_masks(masks_block, enlargement_factor, gaussian_cutoff=config.mask_refinement_cutoff)

                # Add them to the whole image
                masks_count = _add_segmentation_crop(masks_output, z_start, y_start, x_start, masks_block,
                                                     masks_count=masks_count)

    return masks_output


def _add_segmentation_crop(output_array: ndarray, z_start: float, y_start: float, x_start: float, cropped: ndarray, masks_count: int) -> int:
    # Remove masks touching the left, right, top or bottom border (but not the front and back borders)
    border_mask = numpy.full_like(cropped, fill_value=True, dtype=bool)
    border_mask[:, 0, :] = False
    border_mask[:, -1, :] = False
    border_mask[:, :, 0] = False
    border_mask[:, :, -1] = False
    skimage.segmentation.clear_border(cropped, mask=border_mask, out=cropped)

    new_masks = cropped.max()

    cropped[cropped > 0] += masks_count

    masks_count += new_masks

    z_start = int(z_start)
    y_start = int(y_start)
    x_start = int(x_start)

    z_size = min(output_array.shape[0] - z_start, cropped.shape[0])
    y_size = min(output_array.shape[1] - y_start, cropped.shape[1])
    x_size = min(output_array.shape[2] - x_start, cropped.shape[2])
    output_array[z_start:z_start + z_size, y_start:y_start + y_size, x_start:x_start + x_size][cropped > 0] = cropped[:z_size, :y_size, :x_size][cropped > 0]
    return masks_count


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