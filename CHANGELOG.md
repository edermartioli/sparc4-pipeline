Version 0.1.0 (2023-XX-XX)
------------------------

- First initial stable release.
- Included pip instalation using pyproject.toml
- Reorganized layout using `sparc4` module. All library routines goes there.

Version 0.1.1 (2024-04-10)
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

Version 0.1.2 (2024-04-25)
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
- Fixed bug to adapt pipeline to a change in the CALW key value from OFF->None


Versions 0.1.3, 0.1.4, 0.1.5 (2024-06-03)
------------------------
- Bug fixes
- fixed index in Simbad results for target list
- Cast WPPOS value (int) into string to mask table for flat per waveplate position
- Initializing zero=None and loading this initial value that was not recognized because it was not a float
- Crash on image proc missing catalog. The bug was resolved by removing the reference catalog that was created with a different name when there were duplicate entries. Some other bugs were fixed in creating and reading catalogs.
- Fixed bug that was generating an error when there were no images to be reduced
- Divided flux by exptime in new photometry
- Included new method "BY_SIMILARITY" to match frames for stack 
- Fixed bugs in flat correction per wp position
- Included plot_polarimetry_map routine 
- Included plot routines for all 4 channels together for both sci images and lightcurves
- Added PSF fit (Moffat or Gaussian) to obtain more reliable value of the FWHM
- Fixed bugs with background subtraction for selection of images for stack using method "by similarity" 
- Included plot to file capability
- Implemented MEM_CACHE_FOLDER
- Fixed bug in flat per WPPOS
- Fixed a bug in linux for wppos matching in astropy table
- Updated polarimetry plots
- Implemented try/except statements in plots for stability
- Changed parameter ZERO_OF_WAVEPLATE from single float to array of 4 floats, to handle different values between channels
- Implemented fixed number of images per polar seq
- Major changes to implement support for Solar System objects
- Included new tools to plot individual frames and make an animated gif

Version 0.1.6 (2025-02-26)
------------------------
- Fixed bug in get_polarimetry_results() as proposed by Claudia and Fernando
- Updated jupyter notebook for measurement of the zero of waveplate
- Fixed bug in polar time series
- Implemented normalization constant k in get_polarimetry_results() as proposed by Claudia and Ana
- Updated jupyter notebooks to use a maximum aperture of 20 pixels. 
- Fixed bug in the calculation of polarimetric RMS and circular polarization
- Added new measured values for the ZERO_OF_WAVEPLATE
- Updated RMS calculation in polarimetry
- Fixed bug in the sparc4_focus routine
- Implemented new calculation of FWHM
- Fixed inconsistency in the indexation of sources between apertures. 
- Fixed bug in the variable norm being called out of context 

Version 0.1.7 (2025-07-26)
------------------------
- Updated starfind sharp_limit=(0.2, 3.0)
- Fixed bug in casting WPPOS to integer
- Changed identification of sources in astrometry using the new pipeline function utils.gaiadr3_query() instead of twirl
- Fixed bugs in the indexing of SS objects  
- Fixed bug in the reduction of SS objects in PHOT mode
- Fixed bug in plotting 
