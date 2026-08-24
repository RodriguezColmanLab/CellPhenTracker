# Recording intensities
Once you've chosen your segmentation approach, use `Intensity` -> `Record intensities` and select the appropriate option. You will be taken to a new screen. In this screen, in the Parameters menu you can set all required parameters, like in which image channel you want to measure, and decide under which name the intensities are stored. You can have multiple sets of intensities stored, as long as you give them different names.

For the option to record intensities from a pre-existing segmentation, you need to make sure that is segmentation is loaded. In OrganoidTracker, you can (from the main screen) add extra sets of images as extra image channels from the `Edit` menu.

Once you are happy with your chosen parameters, you can use `Edit`-> `Record intensities` from the intensity recording screen to record intensities from all time points. CellPhenTracker will then load all images one by one and record the intensities as position metadata. Once the intensities are recorded, you can exit the screen and go back to the main screen of OrganoidTracker.

[⮩ **Background corrections**](background_corrections.md)  
[⮩ **Normalizations**](normalizing_intensities.md)  
[⮩ **Ratiometric intensities**](ratiometric_intensities.md)
