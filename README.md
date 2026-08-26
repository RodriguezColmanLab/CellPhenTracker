# CellPhenTracker

**[View documentation](https://rodriguezcolmanlab.github.io/CellPhenTracker/)**

Collection of plugins for OrganoidTracker to measure fluorescence. Integrates with existing cell tracking data, and allows you to measure intensities in a variety of ways, normalize them, and plot them.

## Features

### Ways of measuring intensity:

* Using a circle or sphere of a set radius around existing cell positions.
* Using a vertex model: every pixel is assigned to the closest cell position.
* Using an existing segmentation image, obtained by some external program.
* Using existing metadata, obtained by some other plugin or program.
* Using the built-in scaled Cellpose-SAM-based segmentation method.
* Using the built-in intensity threshold-based segmentation method.

You can store multiple intensities, all under their own name.

### Ways of normalizing intensity:

* Multiply all intensities with a single factor so that the median is 1.
* Do a background correction: the background per pixel is set such that the lowest intensity is 0.
* Do a Z correction or time correction: for every Z-layer or time frame, all intensities must have a median of 1.
* All normalizations are stored separately from the raw intensities, and can be undone at any moment.

### Ways of plotting intensities:

* Plot the intensities over time to check for bleaching.
* Plot the intensities by Z-layer to check for scattering.
* Plot the intensities by cell cycle to check for cell cycle effects.
* Plot the intensities in color on top of the image, to check for all kinds of aberrations.
* Plot a lineage tree colored by intensity.
* Plot the intensities for a single selected cell (or multiple) over time.

### Exports:

* CSV/TSV file with time points in rows, cells in columns.
* CSV/TSV file with cells at a single time point in rows, and intensities in columns.

## Installation and usage

First, make sure you have installed OrganoidTracker, and have a dataset & images loaded where you want to measure intensities. Then, open OrganoidTracker, and use `File` -> `Install new plugin`. Click on one of the folders that appears in the submenu, or add a new folder for plugins. Then, place all `plugin_`-files from this repository ([download them here](https://github.com/RodriguezColmanLab/CellPhenTracker/archive/refs/heads/main.zip)) in that folder. Then, back in OrganoidTracker, use `File` -> `Reload all plugins`.

Now a new menu will appear: the `Intensity` menu. From this menu, you can first record the intensities, then normalize them, and then plot them.

## Documentation

Please see the [CellPhenTracker documentation](https://rodriguezcolmanlab.github.io/CellPhenTracker/). The documentation provides detailed information on how to use the software, including segmentation, recording intensities, background corrections, normalizing intensities, and ratiometric intensities..