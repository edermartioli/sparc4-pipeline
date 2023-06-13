# -----------------------------------------------------------------------------
#   define SPARC4 pipeline parameters
# -----------------------------------------------------------------------------
def load_sparc4_parameters() :
    
    #initialize parameters dictionary
    p = {}
    
    #### DIRECTORIES #####
    p['ROOTDATADIR'] = "/Volumes/Samsung_T5/Data/SPARC4/minidata"
    p['ROOTREDUCEDIR'] = "/Volumes/Samsung_T5/Data/SPARC4/minidata/reduced"

    # define SPARC4 channel numbers
    p['CHANNELS'] = [1,2,3,4]
    # define SPARC4 channel labels
    p['CHANNEL_LABELS'] = ['g','r','i','z']

    #### CALIBRATIONS #####
    # wild cards to identify calibration images
    p['CALIB_WILD_CARDS'] = ['*.fits']
    # Method to combine calibration images
    p['CALIB_IMCOMBINE_METHOD'] = 'median'
    # Number of sigmas to clip if using method==mean
    #p['NSIGMA_IMCOMBINE_METHOD'] = 5
    # Value of obstype keyword used to identify bias images
    p['BIAS_OBSTYPE_KEYVALUE'] = 'ZERO'
    # Value of obstype keyword used to identify flat images
    p['FLAT_OBSTYPE_KEYVALUE'] = 'FLAT'
    #p['FLAT_OBSTYPE_KEYVALUE'] = 'DFLAT'
    # Value of obstype keyword used to identify focus images
    p['FOCUS_OBSTYPE_KEYVALUE'] = 'FOCUS'
    # Value of obstype keyword used to identify dark images
    p['DARK_OBSTYPE_KEYVALUE'] = 'DARK'
    # Value of obstype keyword used to identify object images
    p['OBJECT_OBSTYPE_KEYVALUE'] = 'OBJECT'
    
    # Value of INSTMODE keyword used to identify photometric instrument mode
    p['INSTMODE_PHOTOMETRY_KEYVALUE'] = 'PHOT'
    # Value of INSTMODE keyword used to identify polarimetric instrument mode
    p['INSTMODE_POLARIMETRY_KEYVALUE'] = 'POLAR'

    # set maximum number of science frames for each reduction loop
    # it avoids memory issues for long lists
    p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] = 100
    #-------------------------------------
    
    #### SCIENCE DATA #####
    # time keyword in header
    p['TIME_KEY'] = 'DATE-OBS'
    # wild card to identify object images
    p['OBJECT_WILD_CARDS'] = ['*.fits']
    # FITS image extension where science data is located
    p['SCI_EXT'] = 0
    # index of reference image
    p['REF_IMAGE_INDEX'] = 0
    # algorithm to calculate shift: 'cross-correlation' or 'asterism-matching'
    p['SHIFT_ALGORITHM'] = 'asterism-matching'
    #p['SHIFT_ALGORITHM'] = 'cross-correlation'
    ### STACK ###
    # method to select files for stack
    p['METHOD_TO_SELECT_FILES_FOR_STACK'] = 'MAX_FLUXES' # 'MAX_FLUXES' or 'MIN_SHIFTS'
    # stack method
    p['SCI_STACK_METHOD'] = 'median'
    # define number of files for stack
    p['NFILES_FOR_STACK'] = 10
    # define saturation limit
    p['SATURATION_LIMIT'] = 32000
    #-------------------------------------
    
    #### PHOTOMETRY ####
    # whether or not to use multiple apertures
    p['MULTI_APERTURES'] = True
    
    # Define N for automatic calculation of aperture for photometry, where APER_RADIUS = N X FWHM
    p['PHOT_APERTURE_N_X_FWHM'] = 1.5
    p['PHOT_SKYINNER_N_X_FWHM'] = 4.0
    p['PHOT_SKYOUTER_N_X_FWHM'] = 10.0

    # Define aperture size for a fixed aperture when required
    p['PHOT_FIXED_APERTURE'] = 10
    # Define a fixed sky annulus (inner radius, outer radius) in units of pixel
    p['PHOT_FIXED_R_ANNULUS'] = (25, 50)
    
    # Define minimum offset (in pixels) between source aperture radius and sky aperture inner radius
    p['PHOT_MIN_OFFSET_FOR_SKYINNERRADIUS'] = 2
    # Define minimum offset (in pixels) between sky aperture inner and outer radius
    p['PHOT_MIN_OFFSET_FOR_SKYOUTERRADIUS'] = 10

    # Define a list of aperture radii to perform photometry
    p['PHOT_APERTURES'] = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    
    # Threshold in number of sigmas to detect sources, where sigma = background error
    p['PHOT_THRESHOLD'] = 100
    #-------------------------------------

    #### POLARIMETRY ####
    # [S]outh and [N]orth polarimetric beams
    p['CATALOG_BEAM_IDS'] = ["S","N"]
    # tolerance for matching pairs in polarimetric images (units of pixels)
    p['MATCH_PAIRS_TOLERANCE'] = 3.0
    # Set angular sampling of the model in units of degrees
    p['POS_MODEL_SAMPLING'] = 1.0

    # set minimum aperture (pixels) to search for best polar results
    p['MIN_APERTURE_FOR_POLARIMETRY'] = 4
    # set maximum aperture (pixels) to search for best polar results
    p['MAX_APERTURE_FOR_POLARIMETRY'] = 12

    # set aperture index (aperture radius = index + 2) to calculate photometry in polar data
    p['APERTURE_INDEX_FOR_PHOTOMETRY_IN_POLAR'] = 8

    # Set zero of polarimetry calibrated from standards
    p['ZERO_OF_WAVEPLATE'] = 90.5
    
    # Set maximum gap between image indices to break polar sequences
    p['MAX_INDEX_GAP_TO_BREAK_POL_SEQS'] = 1
    # Set maximum gap between image times to break polar sequences
    p['MAX_TIME_GAP_TO_BREAK_POL_SEQS'] = 0.04166
    #-------------------------------------

    #### ASTROMETRY ####
    p['PLATE_SCALE'] = 0.335  # ARCSEC/PIXEL
    #p['PLATE_SCALE'] = 0.18  # ARCSEC/PIXEL 
    p['ASTROM_REF_IMGS'] = ['/Volumes/Samsung_T5/sparc4-pipeline/calibdb/20230503_s4c1_CR1_astrometryRef_stack.fits',
                        '/Volumes/Samsung_T5/sparc4-pipeline/calibdb/20230503_s4c2_CR1_astrometryRef_stack.fits',
                        '/Volumes/Samsung_T5/sparc4-pipeline/calibdb/20230503_s4c3_CR1_astrometryRef_stack.fits',
                        '/Volumes/Samsung_T5/sparc4-pipeline/calibdb/20230503_s4c4_CR1_astrometryRef_stack.fits']
    p['TWEAK_ORDER'] = 3  # order of polynomial to fit astrometry solution
    p['SEARCH_RADIUS'] =  0.1  # radius to define the range of solutions in units of degree
    #-------------------------------------


    #### TIME SERIES ####
    #p['TIME_KEYWORD_IN_PROC'] = 'DATE-OBS'
    #p['TIME_FORMAT_IN_PROC'] = 'isot'
    p['TIME_KEYWORD_IN_PROC'] = 'BJD'
    p['TIME_FORMAT_IN_PROC'] = 'jd'

    p['PHOT_REF_CATALOG_NAME'] = "CATALOG_PHOT_AP010"
    p['POLAR_REF_CATALOG_NAME'] = "CATALOG_POLAR_N_AP010"
    #-------------------------------------

    #### GENERAL PROCESSING #####
    # whether or not to use memmap backend
    p['USE_MEMMAP'] = False
    #-------------------------------------
    
    return p

