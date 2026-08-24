# Normalizing intensities
Normalizing intensities is done from the `Intensity` -> `Normalize intensities` menu from the main OrganoidTracker screen. Normalization essentially calculates one multiplication factor per intensity, such that the median of the intensities of all cells is equal to 1. You can normalize for the whole experiment, per time point or per Z-plane.

Note that normalizations don't stack: if you select a different normalization, the previous one is removed. You can have different normalizations for different intensities, though.

Also note that normalizations don't modify your stored intensity values. They only add some metadata to the tracking data (in `experiment.global_data`), which OrganoidTracker then uses to calculate the actual intensity value. The advantage of this method is that you can later change the normalization.

If you change the [background correction](background_corrections.md), any normalization is removed, and you will have to redo the normalization. This is because the background correction changes the intensity values, and thus the normalization factor needs to be recalculated.
