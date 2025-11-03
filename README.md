# CellPhenTracker

Collection of plugins for OrganoidTracker to measure fluorescent cells.

The idea is that you already have a set of positions from OrganoidTracker, and that you can then measure an intensity for each of these positions.

## Features

Ways of measuring intensity:

* Using a circle or sphere of a set radius.
* Using a vertex model: every pixel is assigned to the closest cell position.
* Using an existing segmentation image, obtained by some external program.
* Using existing metadata, obtained by some other plugin or program.
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

## Installation and usage

First, make sure you have installed OrganoidTracker, and have a dataset & images loaded where you want to measure intensities. Then, open OrganoidTracker, and use `File` -> `Install new plugin`. Click on one of the folders that appears in the submenu, or add a new folder for plugins. Then, place all `plugin_`-files from this repository ([download them here](https://github.com/RodriguezColmanLab/CellPhenTracker/archive/refs/heads/main.zip)) in that folder. Then, back in OrganoidTracker, use `File` -> `Reload all plugins`.

Now a new menu will appear: the `Intensity` menu. From this menu, you can first record the intensities, then normalize them, and then plot them.

# Recording intensities
If you record an intensity, you will be taken to a new screen. In this screen, in the Parameters menu you can set all required parameters, like in which image channel you want to measure, and decide under which name the intensities are stored. You can have multiple sets of intensities stored, as long as you give them different names.

For the option to record intensities from a pre-existing segmentation, you need to make sure that is segmentation is loaded. In OrganoidTracker, you can (from the main screen) add extra sets of images as extra image channels from the `Edit` menu.

Once you are happy with your chosen parameters, you can use `Edit`-> `Record intensities` from the intensity recording screen to record intensities from all time points. CellPhenTracker will then load all images one by one and record the intensities as position metadata. Once the intensities are recorded, you can exit the screen and go back to the main screen of OrganoidTracker.

# Normalizing intensities
Normalizing intensities is done from the `Intensity` menu from the main OrganoidTracker screen.

Note that normalizations don't stack: if you select a different normalization, the previous one is removed. You can have different normalizations for different intensities, though.

Also note that normalizations don't modify your stored intensity values. They only add some metadata to the tracking data (in `experiment.global_data`), which OrganoidTracker then uses to calculate the actual intensity value. The advantage of this method is that you can later on change the normalization.

