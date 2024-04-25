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

Version 1.2 (2024-04-25)
------------------------

- New text file containing a "night report"
- New CSV format for the database file, allowing it to be opened as an Excel spreadsheet, for example
- Introduction of a timeout in the aperture photometry to prevent the system from getting stuck when PhotUtils enters a very long loop.
- Correction of some bugs that could freeze the polarimetry
- Introduction of logging: all messages are now written to the terminal and saved in log files with timestamps.
- New script sparc4_queue.py to reduce multiple nights at once.
- Since the Gaia server was down for a few days, I took the opportunity to implement astrometry using sources from the UCAC catalog obtained through Vizier. It is even possible to switch catalogs through the parameter file, but my tests always worked well with UCAC and not so well with other catalogs.
- Identification of the main target by searching for the OBJECT ID or by coordinates in SIMBAD. When there is an identification, the source is added to the catalogs.
- Correction of other bugs.
- Updated notebooks
- New notebook to reduce the night of 20230605, which was the only one missing to complete the examples for the three nights of the minidata.
- Updated user manual
- Updated GitHub front page
- Option "NIGTHS_INSIDE_CHANNELS_DIR" is now separated for raw and reduced data

