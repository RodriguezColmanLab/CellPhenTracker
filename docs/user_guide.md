# User guide

Measuring intensities in images is a common task in image analysis, especially in biological research. This guide will walk you through the steps necessary to accurately measure the intensity of cells or other structures in your images.

## Segmentation
The first step is to segment the cells in your images. Segmentation is the process of identifying and separating individual cells from the background and other structures in the image. It is not possible to skip this step, although you can use a simple pseudo-segmentation like measuring in a circle around the cell's center.

[⮩ **Segmentation**](segmentation.md)

## Recording intensities
Once you have segmented the cells, you can record the intensities of the cells. This involves measuring the pixel values within the segmented regions and storing them for further analysis.

[⮩ **Recording intensities**](recording_intensities.md)

## Background corrections
If your images have a nonzero background, it is important to perform background corrections to ensure accurate intensity measurements. This step involves subtracting the background intensity from the measured intensities of the cells.

[⮩ **Background corrections**](background_corrections.md)

## Normalizing intensities
Normalizing intensities is an important step to account for variations in imaging conditions across samples. This process involves adjusting the measured intensities to a common scale, allowing for meaningful comparisons between different images or experiments.

[⮩ **Normalizing intensities**](normalizing_intensities.md)

```{eval-rst}
.. Hidden TOCs

.. toctree::
    :maxdepth: 2
    :hidden:

    segmentation
    recording_intensities
    background_corrections
    normalizing_intensities
    ratiometric_intensities
   
```
