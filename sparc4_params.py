# -----------------------------------------------------------------------------
#   define SPARC4 pipeline parameters
# -----------------------------------------------------------------------------
def load_sparc4_parameters() :
    
    #initialize parameters dictionary
    p = {}
    
    #### DIRECTORIES #####
    p['ROOTDATADIR'] = "/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/"
    p['ROOTREDUCEDIR'] = "/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced"
    # define SPARC4 channel numbers
    p['CHANNELS'] = [1,2,3,4]
    # define SPARC4 channel labels
    p['CHANNEL_LABELS'] = ['g','r','i','z']

    #### CALIBRATIONS #####
    # wild cards to identify calibration images
    p['CALIB_WILD_CARDS'] = ['*.fits']
    # Method to combine calibration images
    p['CALIB_IMCOMBINE_METHOD'] = 'median'
    # Value of obstype keyword used to identify bias images
    p['BIAS_OBSTYPE_KEYVALUE'] = 'ZERO'
    # Value of obstype keyword used to identify flat images
    p['FLAT_OBSTYPE_KEYVALUE'] = 'FLAT'
    # Value of obstype keyword used to identify object images
    p['OBJECT_OBSTYPE_KEYVALUE'] = 'OBJECT'
    #-------------------------------------
    
    #### SCIENCE DATA #####
    # wild card to identify object images
    p['OBJECT_WILD_CARDS'] = ['*.fits']
    # FITS image extension where science data is located
    p['SCI_EXT'] = 0
    # index of reference image
    p['REF_IMAGE_INDEX'] = 0
    # algorithm to calculate shift: 'cross-correlation' or 'asterism-matching'
    #p['SHIFT_ALGORITHM'] = 'asterism-matching'
    p['SHIFT_ALGORITHM'] = 'cross-correlation'
    ### STACK ###
    # method to select files for stack
    p['METHOD_TO_SELECT_FILES_FOR_STACK'] = 'MAX_FLUXES' # 'MAX_FLUXES' or 'MIN_SHIFTS'
    # stack method
    p['SCI_STACK_METHOD'] = 'median'
    # define number of files for stack
    p['NFILES_FOR_STACK'] = 30
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
    p['PHOT_THRESHOLD'] = 10
    #-------------------------------------

    #### POLARIMETRY ####
    # [S]outh and [N]orth polarimetric beams
    p['CATALOG_BEAM_IDS'] = ["S","N"]
    # tolerance for matching pairs in polarimetric images (units of pixels)
    p['MATCH_PAIRS_TOLERANCE'] = 3.0
    # Set angular sampling of the model in units of degrees
    p['POS_MODEL_SAMPLING'] = 1.0

    
    #-------------------------------------

    #### ASTROMETRY ####
    p['PLATE_SCALE'] = 0.33  # ARCSEC/PIXEL
    p['ASTROM_REF_IMGS'] = ["/Volumes/Samsung_T5/sparc4-pipeline/calibdb/20221104_s4c1_CR7_stack.fits",
                       "/Volumes/Samsung_T5/sparc4-pipeline/calibdb/20221104_s4c2_CR7_stack.fits",
                       "/Volumes/Samsung_T5/sparc4-pipeline/calibdb/20221104_s4c3_CR7_stack.fits",
                       "/Volumes/Samsung_T5/sparc4-pipeline/calibdb/20221104_s4c4_CR7_stack.fits"]

    #-------------------------------------

    #### GENERAL PROCESSING #####
    # whether or not to use memmap backend
    p['USE_MEMMAP'] = False
    #-------------------------------------
    
    return p
