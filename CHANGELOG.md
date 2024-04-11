Version 1.0 (2023-XX-XX)
------------------------

- First initial stable release.
- Included pip instalation using pyproject.toml
- Reorganized layout using `sparc4` module. All library routines goes there.


Version 1.1 (2024-04-10)
------------------------

- Allows modifying the directory tree with the channels inside the night directory.
- New option to select the first images for stacking.
- Allows inputting parameter file to override default file.
- Allows reduction without Bias and/or Flat.
- Allows truncating the number of bias.
- Includes flat selection by POLAR/PHOT mode, but uses the existing flat if there is no match.
- Allows inputting a target list to force the existence of these objects in the catalogs.
- Static mode implemented in polarimetry.
- Flats usage implemented by blade position.
- plot_light_curve() function transformed into an analysis tool. Allows saving the analysis to a file.