# CellPhenTracker documentation

Welcome to the CellPhenTracker documentation! This documentation provides detailed information on how to use the CellPhenTracker software for measuring intensities in images of cells. The documentation covers various topics, including segmentation, recording intensities, background corrections, normalizing intensities, and ratiometric intensities.

## Installation
To install CellPhenTracker, please follow the instructions provided in the [installation guide](installation.md). Make sure you have OrganoidTracker installed and set up before proceeding with the installation of CellPhenTracker.

[⮩ **Installation**](installation.md)


## Getting started
If you want to start measuring intensities, the first step is to segment the cells in your images.

[⮩ **Segmentation**](segmentation.md)


## Features

Ways of measuring intensity:

* Using a circle or sphere of a set radius.
* Using a vertex model: every pixel is assigned to the closest cell position.
* Using an existing segmentation image, obtained by some external program.
* Using existing metadata, obtained by some other plugin or program.
* Using the built-in scaled Cellpose-SAM-based segmentation method.
* You can store multiple intensities, all under their own name.

Ways of normalizing intensity:

* Multiply all intensities with a single factor so that the median is 1.
* Do a background correction: the background per pixel is set such that the lowest intensity is 0.
* Do a Z correction or time correction: for every Z-layer or time frame, all intensities must have a median of 1.
* All normalizations are stored separately from the raw intensities, and can be undone at any moment.

Ways of plotting intensities:

* Plot the intensities over time to check for bleaching.
* Plot the intensities by Z-layer to check for scattering.
* Plot the intensities by cell cycle to check for cell cycle effects.
* Plot the intensities in color on top of the image, to check for all kinds of aberrations.
* Plot a lineage tree colored by intensity.
* Plot the intensities for a single selected cell (or multiple) over time.

Exports:

* CSV/TSV file with time points in rows, cells in columns.
* CSV/TSV file with cells at a single time point in rows, and intensities in columns.


## Development
CellPhenTracker is developed by the [Rodríguez Colman Lab](https://rodriguezcolmanlab.org/). If you have any questions or need support, please contact us through our website.

[![RCLab Logo](images/rclab_logo.png)](https://rodriguezcolmanlab.org/)

If you used CellPhenTracker in your research, please cite the original paper in which CellPhenTracker was used:

> NTB Nguyen, S Gevers, RNU Kok, LM Burgering, H Neikes, N Akkerman, …, MJ Rodríguez Colman [Lactate controls cancer stemness and plasticity through epigenetic ](https://doi.org/10.1016/j.cmet.2025.01.002) **Cell Metabolism** (2025)



```{eval-rst}
.. Hidden TOCs

.. toctree::
   :maxdepth: 3
   :hidden:
   :includehidden:

   installation
   user_guide

```
