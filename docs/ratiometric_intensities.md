# Ratiometric intensities
If you have two sets of intensities, you can also calculate a ratiometric intensity. This could be used for normalization purposes (like normalizing a reporter by the H2B signal), or for a ratiometric reporter (like a FRET signal). First measure both intensities separately (see above). Then, use `Intensity` -> `Ratiometric intensity` to define a new ratiometric intensity. You will be prompted to select the numerator and denominator, as well as a name.

Note that ratios are calculated after any normalizations and background corrections are applied to the original intensities. You cannot perform normalization on the calculated ratios.

Like normalizations and background corrections, only the metadata is actually stored (in `experiment.global_data`). The actual intensities are calculated on the fly when you plot them. As a result, if you measure the original intensities differently, or change the normalization, the ratiometric intensity will automatically update.
