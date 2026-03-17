# CellPhenTracker

Collection of plugins for OrganoidTracker to measure fluorescent cells.

The idea is that you already have a set of positions from OrganoidTracker, and that you can then measure an intensity for each of these positions.

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

## Installation and usage

First, make sure you have installed OrganoidTracker, and have a dataset & images loaded where you want to measure intensities. Then, open OrganoidTracker, and use `File` -> `Install new plugin`. Click on one of the folders that appears in the submenu, or add a new folder for plugins. Then, place all `plugin_`-files from this repository ([download them here](https://github.com/RodriguezColmanLab/CellPhenTracker/archive/refs/heads/main.zip)) in that folder. Then, back in OrganoidTracker, use `File` -> `Reload all plugins`.

Now a new menu will appear: the `Intensity` menu. From this menu, you can first record the intensities, then normalize them, and then plot them.

# Segmentation
The first step in measuring intensities, is deciding which pixel belong to which cell. In other words: you need to segment your images.

We will assume you have cell tracking data available. In the simplest case, you just measure intensities in a circle or sphere around the cell position. A little more advanced method is to assign each pixel to the closest cell position, up to a certain distance. This is called a vertex model, and is also implemented in CellPhenTracker. Both methods are available in the `Intensity` -> `Record intensities` menu, and are easy to use (although the vertex model is slower to compute).

A more sophisticated approach is to use an actual image-based segmentation method. You can use any external program for this, and load the segmentation into OrganoidTracker (`Edit` -> `Append image channel...`). CellPhenTracker also has a built-in method to segment your images, based on Cellpose-SAM. Your images will by default be downscaled to 1 micron per pixel, which makes the segmentation a lot faster and often also more robust, depending on your images. The resulting segmentations will then be upscaled using a Gaussian blur. This method is available as `Tools` -> `Segment with scaled Cellpose-SAM model`. See below for more details.

# Recording intensities
Once you've chosen your segmentation approach, use `Intensity` -> `Record intensities` and select the appropriate option. You will be taken to a new screen. In this screen, in the Parameters menu you can set all required parameters, like in which image channel you want to measure, and decide under which name the intensities are stored. You can have multiple sets of intensities stored, as long as you give them different names.

For the option to record intensities from a pre-existing segmentation, you need to make sure that is segmentation is loaded. In OrganoidTracker, you can (from the main screen) add extra sets of images as extra image channels from the `Edit` menu.

Once you are happy with your chosen parameters, you can use `Edit`-> `Record intensities` from the intensity recording screen to record intensities from all time points. CellPhenTracker will then load all images one by one and record the intensities as position metadata. Once the intensities are recorded, you can exit the screen and go back to the main screen of OrganoidTracker.

# Normalizing intensities
Normalizing intensities is done from the `Intensity` menu from the main OrganoidTracker screen.

Note that normalizations don't stack: if you select a different normalization, the previous one is removed. You can have different normalizations for different intensities, though.

Also note that normalizations don't modify your stored intensity values. They only add some metadata to the tracking data (in `experiment.global_data`), which OrganoidTracker then uses to calculate the actual intensity value. The advantage of this method is that you can later change the normalization.

# Background correction
CellPhenTracker has a very basic background correction built-in. It assumes a uniform background for each experiment, across all time points and Z-planes. It takes the darkest measured intensity value across all positions, time points and Z-planes, and assumes that this intensity corresponds to the background. Then, from this darkest intensity it calculates the background per pixel, and subtracts this from all intensities. This way, the lowest intensity becomes 0.

If you have a group of cells that have zero signal for the reporter, then this method works out of the box. If you don't have such a group of cells, then you should on purpose define some fake positions in the background, and include those in the intensity measurement. This way, you will have some very low intensity values that will be used to define the background.

# Ratiometric intensities
If you have two sets of intensities, you can also calculate a ratiometric intensity. This could be used for normalization purposes (like normalizing a reporter by the H2B signal), or for a ratiometric reporter (like a FRET signal). First measure both intensities separately (see above). Then, use `Intensity` -> `Ratiometric intensity` to define a new ratiometric intensity. You will be prompted to select the numerator and denominator, as well as a name.

Like normalizations and background corrections, only the metadata is actually stored (in `experiment.global_data`). The actual intensities are calculated on the fly when you plot them. As a result, if you measure the original intensities differently, or change the normalization, the ratiometric intensity will automatically update.

# More details on the Cellpose-SAM segmentation method
From the `Tools` -> `Segment with scaled Cellpose-SAM model` menu, you can segment your images using a built-in method based on Cellpose-SAM. It is reasonably general and not too slow, but it does require a GPU. Make sure you are in the channel that you want to segment (like the nuclei), and then use the menu option. It will generate some scripts.

Besides the location of the input/output files, there a few parameters that you can adjust in the generated scripts. These are:

* `image_channel` - the image channel you want to segment. This is automatically set to the channel you had open when generating the scripts, but you can change it here.
* `target_resolution_zyx_um` - resolution your images will be rescaled to for segmentation.
* `min_percentile` and `max_percentile` - intensity scaling of your images for segmentation. Done per time point.
* `mask_refinement_cutoff` - after the initial segmentation, the masks are blown up in size to reach the original resolution. The masks are smoothed during this process. If your masks are too small, decrease this cutoff. If your masks are too big, increase this cutoff.
* `mask_smoothing_factor` - when blowing up the masks, they are smoothed. This parameter controls how much smoothing is applied. If you see artifacts of the lower resolution segmentation in your final masks, increase this factor. If your masks are too round and miss some protrusions, decrease this factor.
