# Normalizing intensities
Normalizing intensities is done from the `Intensity` menu from the main OrganoidTracker screen.

Note that normalizations don't stack: if you select a different normalization, the previous one is removed. You can have different normalizations for different intensities, though.

Also note that normalizations don't modify your stored intensity values. They only add some metadata to the tracking data (in `experiment.global_data`), which OrganoidTracker then uses to calculate the actual intensity value. The advantage of this method is that you can later change the normalization.