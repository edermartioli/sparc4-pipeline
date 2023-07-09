"""
    Created on May 2 2022

    Description: Library of recipes for the SPARC4 pipeline

    @author: Eder Martioli <martioli@iap.fr>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os

import sparc4.products as s4p
import sparc4.product_plots as s4plt
import sparc4.params as params
import sparc4.utils as s4utils
import sparc4.db as s4db

import glob

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import vstack, Table
from astropy.time import Time
from astropy.modeling import models, fitting

from astropop.file_collection import FitsFileGroup
# astropop used modules
from astropop.image import imcombine, processing, imarith
from astropop.photometry import background, starfind
from astropop.photometry.detection import _fwhm_loop
from astropop.math.array import trim_array, xy2r
from astropop.math.physical import QFloat

import numpy as np
import matplotlib.pyplot as plt

from astropop.image.register import register_framedata_list, compute_shift_list
from astropop.photometry import aperture_photometry

from astropop.polarimetry import match_pairs, estimate_dxdy, SLSDualBeamPolarimetry, quarterwave_model, halfwave_model

from uncertainties import ufloat, umath

from copy import deepcopy

from astropy.coordinates import SkyCoord
from astropop.astrometry import solve_astrometry_xy
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from scipy import signal


def init_s4_p(nightdir, datadir="", reducedir="", channels="", print_report=False):
    """ Pipeline module to initialize SPARC4 parameters
    Parameters
    ----------
    nightdir : str
        String to define the night directory name
    datadir : str, optional
        String to define the directory path to the raw data
    reducedir : str, optional
        String to define the directory path to the reduced data
    channels : str, optional
        String to define SPARC4 channels to be reduced, e.g. "1,2"
    print_report : bool, optional
        Boolean to print out report on all existing data for reduction

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # load pipeline parameters
    p = params.load_sparc4_parameters()

    if datadir != "":
        p['ROOTDATADIR'] = datadir

    if reducedir != "":
        p['ROOTREDUCEDIR'] = reducedir

    p['SELECTED_CHANNELS'] = p['CHANNELS']
    if channels != "":
        p['SELECTED_CHANNELS'] = []
        chs = channels.split(",")
        for ch in chs:
            p['SELECTED_CHANNELS'].append(int(ch))

    # if reduced dir doesn't exist create one
    if not os.path.exists(p['ROOTREDUCEDIR']):
        os.mkdir(p['ROOTREDUCEDIR'])

    # organize files to be reduced
    if print_report:
        p = s4utils.identify_files(p, nightdir, print_report=print_report)

    p['data_directories'] = []
    p['ch_reduce_directories'] = []
    p['reduce_directories'] = []
    p['s4db_files'] = []
    p['filelists'] = []

    for j in range(len(p['CHANNELS'])):
        # figure out directory structures
        ch_night_data_dir = '{}/sparc4acs{}/{}/'.format(
            p['ROOTDATADIR'], p['CHANNELS'][j], nightdir)
        ch_reduce_dir = '{}/sparc4acs{}/'.format(
            p['ROOTREDUCEDIR'], p['CHANNELS'][j])
        reduce_dir = '{}/{}/'.format(ch_reduce_dir, nightdir)

        # produce lists of files for all channels
        channel_data_pattern = '{}/*.fits'.format(ch_night_data_dir)
        p['filelists'].append(sorted(glob.glob(channel_data_pattern)))

        p['ch_reduce_directories'].append(ch_reduce_dir)
        p['reduce_directories'].append(reduce_dir)
        p['data_directories'].append(ch_night_data_dir)

        # if reduced dir doesn't exist create one
        if not os.path.exists(ch_reduce_dir):
            os.mkdir(ch_reduce_dir)

        # if reduced dir doesn't exist create one
        if not os.path.exists(reduce_dir):
            os.mkdir(reduce_dir)

        db_file = '{}/{}/{}_sparc4acs{}_db.fits'.format(
            ch_reduce_dir, nightdir, nightdir, p['CHANNELS'][j])

        p['s4db_files'].append(db_file)

    return p


def reduce_sci_data(db, p, channel_index, inst_mode, detector_mode, nightdir, reduce_dir, polar_mode=None, fit_zero=False, detector_mode_key="", obj=None, match_frames=True, force=False, verbose=False, plot=False):
    """ Pipeline module to run the reduction of science data

    Parameters
    ----------
    db : astropy.table
        data base table
    p : dict
        dictionary to store pipeline parameters
    channel_index : int
        SPARC4 channel index
    inst_mode : str, optional
        to select observations of a given instrument mode
    detector_mode : dict
        to select observations of a given detector mode
    nightdir : str
        String to define the night directory name
    reduce_dir : str
        path to the reduce directory
    polar_mode : str, optional
        to select observations of a given polarimetric mode
    fit_zero : bool
        to fit zero of waveplate position angle
    aperture_index : int (optional)
        index to select aperture in catalog. Default is None and it will calculate best aperture
    detector_mode_key : str (optional)
        keyword name of detector mode
    obj : str (optional)
        object name to reduce only data for this object
    match_frames : bool
        match images using stack as reference
    force : bool
        force reduction even if product already exists
    vebose : bool
        turn on verbose
    plot : bool
        turn on plotting

    Returns
    -------
    p : dict
        updated dictionary to store pipeline parameters
    """

    polsuffix = ""
    polarimetry = False
    if inst_mode == p['INSTMODE_POLARIMETRY_KEYVALUE']:
        polarimetry = True
        polsuffix = "_{}_{}".format(inst_mode, polar_mode)

    # get list of objects observed in photometric mode
    objs = s4db.get_targets_observed(
        db, inst_mode=inst_mode, polar_mode=polar_mode, detector_mode=detector_mode)

    if obj != None:
        objs = objs[objs['OBJECT'] == obj]

    # loop over each object to run the reduction
    for k in range(len(objs)):
        obj = objs[k][0]

        # set suffix for output stack filename
        stack_suffix = "{}_s4c{}{}_{}{}".format(
            nightdir, p['CHANNELS'][channel_index], detector_mode_key, obj.replace(" ", ""), polsuffix)

        sci_list = s4db.get_file_list(db, object_id=obj, inst_mode=inst_mode, polar_mode=polar_mode,
                                      obstype=p['OBJECT_OBSTYPE_KEYVALUE'], calwheel_mode=None, detector_mode=detector_mode)

        # run stack and reduce individual science images (produce *_proc.fits)
        p = stack_and_reduce_sci_images(p,
                                        sci_list,
                                        reduce_dir,
                                        ref_img="",
                                        stack_suffix=stack_suffix,
                                        force=force,
                                        match_frames=match_frames,
                                        polarimetry=polarimetry,
                                        verbose=verbose,
                                        plot=plot)

        # set suffix for output time series filename
        ts_suffix = "{}_s4c{}_{}{}".format(
            nightdir, p['CHANNELS'][channel_index], obj.replace(" ", ""), polsuffix)

        ###################################
        ## TIME SERIES FOR PHOTOMETRIC MODE ##
        ###################################
        if inst_mode == p['INSTMODE_PHOTOMETRY_KEYVALUE']:
            # run photometric time series
            phot_ts_product = phot_time_series(p['OBJECT_REDUCED_IMAGES'][1:],
                                               ts_suffix=ts_suffix,
                                               reduce_dir=reduce_dir,
                                               time_key=p['TIME_KEYWORD_IN_PROC'],
                                               time_format=p['TIME_FORMAT_IN_PROC'],
                                               catalog_names=p['PHOT_CATALOG_NAMES_TO_INCLUDE'],
                                               time_span_for_rms=p['TIME_SPAN_FOR_RMS'],
                                               force=force)

            if plot:
                target = 0
                comps = [1, 2, 3, 4]
                # plot light curve
                s4plt.plot_light_curve(phot_ts_product,
                                       target=target,
                                       comps=comps,
                                       nsig=10,
                                       plot_coords=True,
                                       plot_rawmags=True,
                                       plot_sum=True,
                                       plot_comps=True,
                                       catalog_name=p['PHOT_REF_CATALOG_NAME'])

        ##############################################
        ## POLARIMETRY AND TIME SERIES FOR POLARIMETRIC MODE (L2 OR L4) ##
        ##############################################
        elif inst_mode == p['INSTMODE_POLARIMETRY_KEYVALUE']:

            compute_k, zero = True, 0
            wave_plate = 'halfwave'

            if polar_mode == 'L4':
                wave_plate = 'quarterwave'
                compute_k = False
                zero = p['ZERO_OF_WAVEPLATE']

            # divide input list into a many sequences
            pol_sequences = s4utils.select_polar_sequences(
                p['OBJECT_REDUCED_IMAGES'], sortlist=True, verbose=verbose)

            p['PolarProducts'] = []

            for i in range(len(pol_sequences)):

                if len(pol_sequences[i]) == 0:
                    continue

                if verbose:
                    print("Running {} polarimetry for sequence: {} of {}".format(
                        wave_plate, i+1, len(pol_sequences)))
                polarproduct = compute_polarimetry(pol_sequences[i],
                                                   wave_plate=wave_plate,
                                                   base_aperture=p['APERTURE_RADIUS_FOR_PHOTOMETRY_IN_POLAR'],
                                                   compute_k=compute_k,
                                                   fit_zero=fit_zero,
                                                   zero=zero)

                pol_results = get_polarimetry_results(polarproduct,
                                                      source_index=0,
                                                      min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                                      max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                                      plot=plot,
                                                      verbose=verbose)
                p['PolarProducts'].append(polarproduct)

            # create polar time series product
            p['PolarTimeSeriesProduct'] = polar_time_series(p['PolarProducts'],
                                                            reduce_dir=reduce_dir,
                                                            ts_suffix=ts_suffix,
                                                            aperture_radius=p['APERTURE_RADIUS_FOR_PHOTOMETRY_IN_POLAR'],
                                                            min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                                            max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                                            force=force)
        else:
            print("ERROR: instrument mode not supported, exiting ... ")
            exit()

    return p


def run_master_calibration(p, inputlist=[], output="", obstype='bias', data_dir="./", reduce_dir="./", normalize=False, force=False):
    """ Pipeline module to run master calibration

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters

    obstype : str, optional
        String to define the type of observation.
        It accepts the following values: 'bias', 'flat', 'object'.
    data_dir : str, optional
        String to define the directory path to the raw data
    reduce_dir : str, optional
        String to define the directory path to the reduced data
    normalize : bool, optional
        Boolean to decide whether or not to normalize the data
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    obstype_keyvalue = 'UNKNOWN'
    if obstype == 'bias':
        obstype_keyvalue = p['BIAS_OBSTYPE_KEYVALUE']
    elif obstype == 'flat':
        obstype_keyvalue = p['FLAT_OBSTYPE_KEYVALUE']
    elif obstype == 'object':
        obstype_keyvalue = p['OBJECT_OBSTYPE_KEYVALUE']
    else:
        print("obstype={} not recognized, setting to default = {}".format(
            obstype, obstype_keyvalue))

    if output == "":
        output = os.path.join(reduce_dir, "master_{}.fits".format(obstype))

    # set master calib keyword in parameters
    p["master_{}".format(obstype)] = output

    # Skip if product already exists and reduction is not forced
    if os.path.exists(output) and not force:
        return p

    # set method to combine images
    method = p['CALIB_IMCOMBINE_METHOD']

    if inputlist == []:
        # select FITS files in the minidata directory and build database
        main_fg = FitsFileGroup(
            location=data_dir, fits_ext=p['CALIB_WILD_CARDS'], ext=0)
        # print total number of files selected:
        print(f'Total files: {len(main_fg)}')

        # Filter files by header keywords
        filter_fg = main_fg.filtered({'obstype': obstype_keyvalue})
    else:
        filter_fg = FitsFileGroup(files=inputlist)

    # print total number of bias files selected
    print(f'{obstype} files: {len(filter_fg)}')

    # get frames
    frames = list(filter_fg.framedata(
        unit='adu', use_memmap_backend=p['USE_MEMMAP']))

    # extract gain from the first image
    if float(frames[0].header['GAIN']) != 0:
        gain = float(frames[0].header['GAIN'])*u.electron / \
            u.adu  # using quantities is better for safety
    else:
        gain = 3.3*u.electron/u.adu
    print('gain:', gain)

    # Perform gain calibration
    for i, frame in enumerate(frames):
        print(f'processing frame {i+1} of {len(frames)}')
        processing.cosmics_lacosmic(frame, inplace=True)
        processing.gain_correct(frame, gain, inplace=True)

    # combine
    master = imcombine(frames, method=method,
                       use_memmap_backend=p['USE_MEMMAP'])

    # get statistics
    stats = master.statistics()

    norm_mean_value = master.mean()
    print('Normalization mean value:', norm_mean_value)

    data_units = 'electron'

    if normalize:
        master = imarith(master, norm_mean_value, '/')
        data_units = 'dimensionless'

    # write information into an info dict
    info = {'INCOMBME': ('{}'.format(method), 'imcombine method'),
            'INCOMBNI': (len(filter_fg), 'imcombine nimages'),
            'BUNIT': ('{}'.format(data_units), 'data units'),
            'DRSINFO': ('astropop', 'data reduction software'),
            'DRSROUT': ('master image', 'data reduction routine'),
            'NORMALIZ': (normalize, 'normalized master'),
            'NORMMEAN': (norm_mean_value.value, 'normalization mean value in {}'.format(norm_mean_value.unit)),
            'MINVAL': (stats['min'].value, 'minimum value in {}'.format(stats['min'].unit)),
            'MAXVAL': (stats['max'].value, 'maximum value in {}'.format(stats['max'].unit)),
            'MEANVAL': (stats['mean'].value, 'mean value in {}'.format(stats['mean'].unit)),
            'MEDIANVA': (stats['median'].value, 'median value in {}'.format(stats['median'].unit)),
            'STDVALUE': (stats['std'].value, 'standard deviation in {}'.format(stats['std'].unit))
            }

    # get data arrays
    img_data = np.array(master.data)
    err_data = np.array(master.get_uncertainty())
    mask_data = np.array(master.mask)

    # call function masteZero from sparc4_products to generate final product
    mastercal = s4p.masterCalibration(filter_fg.files, img_data=img_data,
                                      err_data=err_data, mask_data=mask_data, info=info, filename=output)

    return p


def old_reduce_science_images(p, inputlist, data_dir="./", reduce_dir="./", force=False, match_frames=False, stack_suffix="", output_stack="", polarimetry=False):
    """ Pipeline module to run the reduction of science images.

         The reduction consists of the following processing steps:
         1. Detect and mask cosmic rays
         2. Perform gain correction, convert units from ADU to electrons
         3. Subtract master bias
         4. Divide by a master flat field
         5. Calculate linear offset between images
         6. Select sub-set of images to calculate a stack
         7. Obtain a catalog of point-like sources from the stack image
         8. Calculate FWHM for all sources and set an aperture for photometry
         9. Perform aperture photometry for all images on all sources in the catalog
         10. Save reduced image, catalog data, and photometry into a S4 product

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters

    data_dir : str, optional
        String to define the directory path to the raw data
    reduce_dir : str, optional
        String to define the directory path to the reduced data
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists
    match_frames : bool, optional
        Boolean to decide whether or not to match frames, usually for
        data taken as time series
    output_stack : str, optional
        String to define the directory path to the output stack file
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # read master calibration files
    try:
        # load bias frame
        bias = s4p.getFrameFromMasterCalibration(p["master_bias"])
    except:
        print("WARNING: failed to read master bias, ignoring ...")

    try:
        # load flat frame
        flat = s4p.getFrameFromMasterCalibration(p["master_flat"])
    except:
        print("WARNING: failed to read master flat, ignoring ...")

    # select FITS files in the minidata directory and build database
    # main_fg = FitsFileGroup(location=data_dir, fits_ext=p['OBJECT_WILD_CARDS'], ext=p['SCI_EXT'])
    obj_fg = FitsFileGroup(files=inputlist)

    # print total number of object files selected
    print(f'OBJECT files: {len(obj_fg)}')

    # set base image as the reference image, which will be replaced later if run registering
    p['REF_IMAGE_INDEX'] = 0
    p['REFERENCE_IMAGE'] = obj_fg.files[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    # set output stack filename
    if output_stack == "":
        output_stack = os.path.join(
            reduce_dir, '{}_stack.fits'.format(stack_suffix))
    p['OBJECT_STACK'] = output_stack

    print("Creating output list of processed science frames ... ")

    p['OBJECT_REDUCED_IMAGES'], obj_red_status = [], []

    for i in range(len(obj_fg.files)):
        # get basename
        basename = os.path.basename(obj_fg.files[i])
        # create output name in the reduced dir
        output = os.path.join(
            reduce_dir, basename.replace(".fits", "_proc.fits"))
        p['OBJECT_REDUCED_IMAGES'].append(output)

        red_status = False
        if not force:
            if os.path.exists(output):
                red_status = True
        obj_red_status.append(red_status)

        print("{} of {} is reduced? {} -> {}".format(i +
              1, len(obj_fg.files), red_status, output))

    if not all(obj_red_status) or force:
        print("Loading science frames to memory ... ")
        # get frames
        frames = list(obj_fg.framedata(
            unit='adu', use_memmap_backend=p['USE_MEMMAP']))

        # extract gain from the first image
        if float(frames[0].header['GAIN']) != 0:
            # using quantities is better for safety
            gain = float(frames[0].header['GAIN'])*u.electron/u.adu
        else:
            gain = 3.3*u.electron/u.adu

        print('gain:', gain)

        # set units of reduced data
        data_units = 'electron'

        # write information into an info dict
        info = {'BUNIT': ('{}'.format(data_units), 'data units'),
                'DRSINFO': ('astropop', 'data reduction software'),
                'DRSROUT': ('science frame', 'data reduction routine'),
                'BIASSUB': (True, 'bias subtracted'),
                'BIASFILE': (p["master_bias"], 'bias file name'),
                'FLATCORR': (True, 'flat corrected'),
                'FLATFILE': (p["master_flat"], 'flat file name')
                }

        print('Calibrating science frames (CR, gain, bias, flat) ... ')

        # Perform calibration
        for i, frame in enumerate(frames):
            print("Calibrating science frame {} of {} : {} ".format(
                i+1, len(frames), os.path.basename(obj_fg.files[i])))
            if not obj_red_status[i] or force:
                processing.cosmics_lacosmic(frame, inplace=True)
                processing.gain_correct(frame, gain, inplace=True)
                processing.subtract_bias(frame, bias, inplace=True)
                processing.flat_correct(frame, flat, inplace=True)
            else:
                pass

        if match_frames:
            print('Calculating offsets and selecting images for stack ... ')
            # run routine to select files that will be used for stack
            p = select_files_for_stack_and_get_shifts(
                p, frames, obj_fg.files, sort_method=p['METHOD_TO_SELECT_FILES_FOR_STACK'])

            info['REFIMG'] = (p['REFERENCE_IMAGE'],
                              "reference image for stack")
            info['NIMGSTCK'] = (p['FINAL_NFILES_FOR_STACK'],
                                "number of images for stack")

            print('Registering science frames ... ')
            # Register images, generate global catalog and generate stack image
            p = run_register_frames(p, frames, obj_fg.files, info,
                                    output_stack=output_stack, force=force, polarimetry=polarimetry)

        # Perform aperture photometry and store reduced data into products
        for i, frame in enumerate(frames):

            info['XSHIFT'] = (0., "register x shift (pixel)")
            info['XSHIFTST'] = ("OK", "x shift status")
            info['YSHIFT'] = (0., "register y shift (pixel)")
            info['YSHIFTST'] = ("OK", "y shift status")
            if match_frames:
                if np.isfinite(p["XSHIFTS"][i]):
                    info['XSHIFT'] = (
                        p["XSHIFTS"][i], "register x shift (pixel)")
                else:
                    info['XSHIFTST'] = ("UNDEFINED", "x shift status")

                if np.isfinite(p["YSHIFTS"][i]):
                    info['YSHIFT'] = (
                        p["YSHIFTS"][i], "register y shift (pixel)")
                else:
                    info['YSHIFTST'] = ("UNDEFINED", "y shift status")

            if not obj_red_status[i] or force:
                # get data arrays
                img_data = np.array(frame.data)
                err_data = np.array(frame.get_uncertainty())
                mask_data = np.array(frame.mask)

                try:
                    # make catalog
                    if match_frames:
                        p, frame_catalogs = build_catalogs(p, img_data, deepcopy(
                            p["CATALOGS"]), xshift=p["XSHIFTS"][i], yshift=p["YSHIFTS"][i], polarimetry=polarimetry)
                    else:
                        p, frame_catalogs = build_catalogs(
                            p, img_data, polarimetry=polarimetry)
                except:
                    print("WARNING: could not build frame catalog")
                    # set local
                    frame_catalogs = []

                print("Saving frame {} of {}:".format(i+1, len(frames)),
                      obj_fg.files[i], '->', p['OBJECT_REDUCED_IMAGES'][i])

                frame_wcs_header = deepcopy(p['WCS_HEADER'])

                if np.isfinite(p["XSHIFTS"][i]):
                    frame_wcs_header['CRPIX1'] = frame_wcs_header['CRPIX1'] + \
                        p["XSHIFTS"][i]
                if np.isfinite(p["YSHIFTS"][i]):
                    frame_wcs_header['CRPIX2'] = frame_wcs_header['CRPIX2'] + \
                        p["YSHIFTS"][i]

                # call function to generate final product
                # for light products
                s4p.scienceImageLightProduct(obj_fg.files[i], img_data=img_data, info=info, catalogs=frame_catalogs, polarimetry=polarimetry,
                                             filename=p['OBJECT_REDUCED_IMAGES'][i], catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=frame_wcs_header, time_key=p["TIME_KEY"])
                # for more complete products with an error and mask extensions
                # s4p.scienceImageProduct(obj_fg.files[i], img_data=img_data, err_data=err_data, mask_data=mask_data, info=info, catalogs=frame_catalogs, polarimetry=polarimetry, filename=p['OBJECT_REDUCED_IMAGES'][i], catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=frame_wcs_header, time_key=p["TIME_KEY"])

    return p


def reduce_science_images(p, inputlist, data_dir="./", reduce_dir="./", match_frames=True, ref_img="", force=False, polarimetry=False, ra="", dec=""):
    """ Pipeline module to run the reduction of science images.

         The reduction consists of the following processing steps:
         1. Detect and mask cosmic rays
         2. Perform gain correction, convert units from ADU to electrons
         3. Subtract master bias
         4. Divide by a master flat field
         5. Calculate linear offset between images
         8. Calculate FWHM for all sources and set an aperture for photometry
         9. Perform aperture photometry for all images on all sources in the catalog
         10. Save reduced image, catalog data, and photometry into a S4 product

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters

    data_dir : str, optional
        String to define the directory path to the raw data
    reduce_dir : str, optional
        String to define the directory path to the reduced data
    match_frames : bool, optional
        Boolean to decide whether or not to match frames, usually for
        data taken as time series
    ref_img : str, optional
        reference image, if not given it selects an image in the sequence
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product
        already exists
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with
        duplicated sources
    ra : str, optional
        string to overwrite header RA (Right Ascension) keyword
    dec : str, optional
        string to overwrite header DEC (Declination) keyword

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # read master calibration files
    try:
        # load bias frame
        bias = s4p.getFrameFromMasterCalibration(p["master_bias"])
    except:
        print("WARNING: failed to read master bias, ignoring ...")

    try:
        # load flat frame
        flat = s4p.getFrameFromMasterCalibration(p["master_flat"])
    except:
        print("WARNING: failed to read master flat, ignoring ...")

    # save original input list of files
    p['INPUT_LIST_OF_FILES'] = deepcopy(inputlist)
    # check whether the input reference image is in the input list
    if ref_img not in inputlist:
        # add ref image to the list
        inputlist = [ref_img] + inputlist

    # make sure to get the correct index for the reference image in the new list
    p['REF_IMAGE_INDEX'] = inputlist.index(p['REFERENCE_IMAGE'])

    # select FITS files in the minidata directory and build database
    # main_fg = FitsFileGroup(location=data_dir, fits_ext=p['OBJECT_WILD_CARDS'], ext=p['SCI_EXT'])
    obj_fg = FitsFileGroup(files=inputlist)

    # print total number of object files selected
    print(f'OBJECT files: {len(obj_fg)}')

    print("Creating output list of processed science frames ... ")

    obj_red_images, obj_red_status = [], []

    for i in range(len(obj_fg.files)):
        # get basename
        basename = os.path.basename(obj_fg.files[i])
        # create output name in the reduced dir
        output = os.path.join(
            reduce_dir, basename.replace(".fits", "_proc.fits"))
        obj_red_images.append(output)

        red_status = False
        if not force:
            if os.path.exists(output):
                red_status = True
        obj_red_status.append(red_status)

        print("{} of {} is reduced? {} -> {}".format(i +
              1, len(obj_fg.files), red_status, output))

    if not all(obj_red_status) or force:
        print("Loading science frames to memory ... ")
        # get frames
        frames = list(obj_fg.framedata(
            unit='adu', use_memmap_backend=p['USE_MEMMAP']))

        # extract gain from the first image
        if float(frames[0].header['GAIN']) != 0:
            # using quantities is better for safety
            gain = float(frames[0].header['GAIN'])*u.electron/u.adu
        else:
            gain = 3.3*u.electron/u.adu

        # print gain value
        print('gain:', gain)

        # set units of reduced data
        data_units = 'electron'

        # write information into an info dict
        info = {'BUNIT': ('{}'.format(data_units), 'data units'),
                'DRSINFO': ('astropop', 'data reduction software'),
                'DRSROUT': ('science frame', 'data reduction routine'),
                'BIASSUB': (True, 'bias subtracted'),
                'BIASFILE': (p["master_bias"], 'bias file name'),
                'FLATCORR': (True, 'flat corrected'),
                'FLATFILE': (p["master_flat"], 'flat file name')
                }

        print('Calibrating science frames (CR, gain, bias, flat) ... ')

        # Perform calibration
        for i, frame in enumerate(frames):
            print("Calibrating science frame {} of {} : {} ".format(
                i+1, len(frames), os.path.basename(obj_fg.files[i])))
            if not obj_red_status[i] or force:
                processing.cosmics_lacosmic(frame, inplace=True)
                processing.gain_correct(frame, gain, inplace=True)
                processing.subtract_bias(frame, bias, inplace=True)
                processing.flat_correct(frame, flat, inplace=True)
            else:
                pass

        print('Calculating offsets ... ')
        p = compute_offsets(p, frames, obj_fg.files, auto_ref_selection=False)

        # write ref image to header
        info['REFIMG'] = (p['REFERENCE_IMAGE'], "reference image")

        # Perform aperture photometry and store reduced data into products
        for i, frame in enumerate(frames):

            info['XSHIFT'] = (0., "register x shift (pixel)")
            info['XSHIFTST'] = ("OK", "x shift status")
            info['YSHIFT'] = (0., "register y shift (pixel)")
            info['YSHIFTST'] = ("OK", "y shift status")
            if match_frames:
                if np.isfinite(p["XSHIFTS"][i]):
                    info['XSHIFT'] = (
                        p["XSHIFTS"][i], "register x shift (pixel)")
                else:
                    info['XSHIFTST'] = ("UNDEFINED", "x shift status")

                if np.isfinite(p["YSHIFTS"][i]):
                    info['YSHIFT'] = (
                        p["YSHIFTS"][i], "register y shift (pixel)")
                else:
                    info['YSHIFTST'] = ("UNDEFINED", "y shift status")

            if not obj_red_status[i] or force:
                # get data arrays
                img_data = np.array(frame.data)
                try:
                    # make catalog
                    if match_frames and "CATALOGS" in p.keys():
                        p, frame_catalogs = build_catalogs(p, img_data, deepcopy(
                            p["CATALOGS"]), xshift=p["XSHIFTS"][i], yshift=p["YSHIFTS"][i], polarimetry=polarimetry)
                    else:
                        p, frame_catalogs = build_catalogs(
                            p, img_data, polarimetry=polarimetry)
                except:
                    print("WARNING: could not build frame catalog.")
                    # set local
                    frame_catalogs = []

                print("Saving frame {} of {}:".format(i+1, len(frames)),
                      obj_fg.files[i], '->', obj_red_images[i])

                frame_wcs_header = deepcopy(p['WCS_HEADER'])

                if np.isfinite(p["XSHIFTS"][i]):
                    frame_wcs_header['CRPIX1'] = frame_wcs_header['CRPIX1'] + \
                        p["XSHIFTS"][i]
                if np.isfinite(p["YSHIFTS"][i]):
                    frame_wcs_header['CRPIX2'] = frame_wcs_header['CRPIX2'] + \
                        p["YSHIFTS"][i]

                # call function to generate final product
                # for light products
                s4p.scienceImageLightProduct(obj_fg.files[i], img_data=img_data, info=info, catalogs=frame_catalogs, polarimetry=polarimetry,
                                             filename=obj_red_images[i], catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=frame_wcs_header, time_key=p["TIME_KEY"], ra=ra, dec=dec)

    if 'OBJECT_REDUCED_IMAGES' not in p.keys():
        p['OBJECT_REDUCED_IMAGES'] = obj_red_images
    else:
        for i in range(len(obj_red_images)):
            if obj_red_images[i] not in p['OBJECT_REDUCED_IMAGES']:
                p['OBJECT_REDUCED_IMAGES'].append(obj_red_images[i])
    return p


def stack_science_images(p, inputlist, reduce_dir="./", force=False, stack_suffix="", output_stack="", polarimetry=False):
    """ Pipeline module to run the stack of science images.

         The reduction consists of the following processing steps:
         1. Detect and mask cosmic rays
         2. Perform gain correction, convert units from ADU to electrons
         3. Subtract master bias
         4. Divide by a master flat field
         5. Calculate linear offset between images
         6. Select sub-set of images to calculate a stack
         7. Obtain a catalog of point-like sources from the stack image
         8. Calculate FWHM for all sources and set an aperture for photometry
         9. Perform aperture photometry for all images on all sources in the catalog
         10. Save reduced image, catalog data, and photometry into a S4 product

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters

    reduce_dir : str, optional
        String to define the directory path to the reduced data
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists
    output_stack : str, optional
        String to define the directory path to the output stack file
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # set output stack filename
    if output_stack == "":
        output_stack = os.path.join(
            reduce_dir, '{}_stack.fits'.format(stack_suffix))
    p['OBJECT_STACK'] = output_stack

    if os.path.exists(p['OBJECT_STACK']) and not force:
        print("There is already a stack image :", p['OBJECT_STACK'])

        stack_hdulist = fits.open(p['OBJECT_STACK'])
        hdr = stack_hdulist[0].header
        p['REFERENCE_IMAGE'] = hdr['REFIMG']
        p['REF_IMAGE_INDEX'] = 0
        for i in range(len(inputlist)):
            if inputlist[i] == p['REFERENCE_IMAGE']:
                p['REF_IMAGE_INDEX'] = i
        p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

        p["CATALOGS"] = s4p.readScienceImagCatalogs(p['OBJECT_STACK'])

        return p

    # read master calibration files
    try:
        # load bias frame
        bias = s4p.getFrameFromMasterCalibration(p["master_bias"])
    except:
        print("WARNING: failed to read master bias, ignoring ...")

    try:
        # load flat frame
        flat = s4p.getFrameFromMasterCalibration(p["master_flat"])
    except:
        print("WARNING: failed to read master flat, ignoring ...")

    # set base image as the reference image, which will be replaced
    # later if run "select_files_for_stack"
    p['REF_IMAGE_INDEX'] = 0
    p['REFERENCE_IMAGE'] = inputlist[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    # first select best files for stack
    p = select_files_for_stack(
        p, inputlist, saturation_limit=p['SATURATION_LIMIT'], imagehdu=0)

    # select FITS files in the minidata directory and build database
    obj_fg = FitsFileGroup(files=p['SELECTED_FILES_FOR_STACK'])

    # print total number of object files selected
    print(f'OBJECT files: {len(obj_fg)}')

    print("Loading science frames to memory ... ")
    # get frames
    frames = list(obj_fg.framedata(
        unit='adu', use_memmap_backend=p['USE_MEMMAP']))

    # extract gain from the first image
    if float(frames[0].header['GAIN']) != 0:
        gain = float(frames[0].header['GAIN'])*u.electron / \
            u.adu  # using quantities is better for safety
    else:
        gain = 3.3*u.electron/u.adu

    print('gain:', gain)

    # set units of reduced data
    data_units = 'electron'

    # write information into an info dict
    info = {'BUNIT': ('{}'.format(data_units), 'data units'),
            'DRSINFO': ('astropop', 'data reduction software'),
            'DRSROUT': ('science frame', 'data reduction routine'),
            'BIASSUB': (True, 'bias subtracted'),
            'BIASFILE': (p["master_bias"], 'bias file name'),
            'FLATCORR': (True, 'flat corrected'),
            'FLATFILE': (p["master_flat"], 'flat file name'),
            'REFIMG': (p['REFERENCE_IMAGE'], 'reference image for stack'),
            'NIMGSTCK': (p['FINAL_NFILES_FOR_STACK'], 'number of images for stack')
            }

    print('Calibrating science frames (CR, gain, bias, flat) ... ')

    # Perform calibration
    for i, frame in enumerate(frames):
        print("Calibrating science frame {} of {} : {} ".format(
            i+1, len(frames), os.path.basename(obj_fg.files[i])))
        processing.cosmics_lacosmic(frame, inplace=True)
        processing.gain_correct(frame, gain, inplace=True)
        processing.subtract_bias(frame, bias, inplace=True)
        processing.flat_correct(frame, flat, inplace=True)

    print('Registering science frames and stacking them ... ')

    p['SELECTED_FILE_INDICES_FOR_STACK'] = np.arange(
        p['FINAL_NFILES_FOR_STACK'])

    # Register images, generate global catalog and generate stack image
    p = run_register_frames(p, frames, obj_fg.files, info,
                            output_stack=output_stack, force=force, polarimetry=polarimetry)

    return p


def select_files_for_stack(p, inputlist, saturation_limit=32768, imagehdu=0):
    """ Pipeline module to select a sub-set of frames for stack

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    inputlist : list of str len=N_sci_images
        list of file paths to the science object image files
    saturation_limit : int or float, optional
        saturation flux limit, images with any max value above or equal this value will be ignored
    imagehdu : int or string, optional
        HDU index/name containing the image data

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """
    peaks = []
    maxflux, ref_img_idx = 0, 0
    for i in range(len(inputlist)):
        img = fits.getdata(inputlist[i], imagehdu)
        keep = img < saturation_limit
        bkg = np.nanmedian(img[keep])
        maximgflux = np.nanmax(img[keep]) - bkg
        if maximgflux > maxflux:
            maxflux = maximgflux
            ref_img_idx = i
        peaks.append(maximgflux)
        # print(i, inputlist[i], '-> ', bkg, (maximgflux - bkg))
    peaks = np.array(peaks)

    p['REF_IMAGE_INDEX'] = ref_img_idx
    p['REFERENCE_IMAGE'] = inputlist[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    # plt.plot(peaks)
    # plt.show()
    print(p['REF_IMAGE_INDEX'], "Reference image: {}".format(p['REFERENCE_IMAGE']))

    sort = np.flip(np.argsort(peaks))

    sorted_files = []
    newsort = []
    # first add reference image
    sorted_files.append(p['REFERENCE_IMAGE'])
    newsort.append(p['REF_IMAGE_INDEX'])
    # then add all valid images to the list of images for stack
    for i in sort:
        if inputlist[i] != p['REFERENCE_IMAGE']:
            sorted_files.append(inputlist[i])
            newsort.append(i)
    newsort = np.array(newsort)

    # Now select up to <N files for stack as defined in the parameters file and save list to the param dict
    if len(sorted_files) > p['NFILES_FOR_STACK']:
        p['SELECTED_FILES_FOR_STACK'] = sorted_files[:p['NFILES_FOR_STACK']]
        # p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort[:p['NFILES_FOR_STACK']]
        p['FINAL_NFILES_FOR_STACK'] = p['NFILES_FOR_STACK']
    else:
        p['FINAL_NFILES_FOR_STACK'] = len(sorted_files)
        p['SELECTED_FILES_FOR_STACK'] = sorted_files
        # p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort

    return p


def compute_offsets(p, frames, obj_files, auto_ref_selection=False):
    """ Pipeline module to select a reference image and
    compute offsets between science images

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    frames : list of frame type len=N_sci_images
        list of frames containing the data of science images
    obj_files : list of str len=N_sci_images
        list of file paths to the science object image files
    auto_ref_selection : bool, optional
        to select a reference image automatically from peak flux

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    if auto_ref_selection:
        peaksnr = []
        for i, frame in enumerate(frames):
            bkg, rms = background(frame.data, global_bkg=False)
            bkgsubimg = frame.data - bkg
            snr = bkgsubimg / rms
            peaksnr.append(np.nanmax(snr))
        peaksnr = np.array(peaksnr, dtype=float)

        p['REF_IMAGE_INDEX'] = np.argmax(peaksnr)
        p['REFERENCE_IMAGE'] = obj_files[p['REF_IMAGE_INDEX']]
        p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    print("Computing offsets with respect to the reference image: index={} -> {}".format(
        p['REF_IMAGE_INDEX'], obj_files[p['REF_IMAGE_INDEX']]))

    # get x and y shifts of all images with respect to the first image
    shift_list = compute_shift_list(
        frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=p['REF_IMAGE_INDEX'], skip_failure=True)

    # store shifts in x and y np arrays
    x, y = np.array([]), np.array([])
    for i in range(len(shift_list)):
        x = np.append(x, shift_list[i][0])
        y = np.append(y, shift_list[i][1])
        # print(i,shift_list[i][0],shift_list[i][1])

    # store shifts
    p["XSHIFTS"] = x
    p["YSHIFTS"] = y

    return p


def select_files_for_stack_and_get_shifts(p, frames, obj_files, sort_method='MAX_FLUXES', correct_shifts=False, plot=False):
    """ Pipeline module to compute offset between all science images and
        select a sub-set of frames for stack

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    frames : list of frame type len=N_sci_images
        list of frames containing the data of science images
    obj_files : list of str len=N_sci_images
        list of file paths to the science object image files
    sort_method : 'str'
        method to sort frames and select best subset.
        possible values:
            'MAX_FLUXES' -> sort by flux in reverse order
            'MIN_SHIFTS' -> sort by offsets
    correct_shifts : bool, optional
        Boolean to decide whether or not to correct shifts
    plot : bool, optional
        Boolean to decide whether or not to plot

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    peaks, meanbkg, peaksnr = [], [], []
    for i, frame in enumerate(frames):
        bkg, rms = background(frame.data, global_bkg=False)
        bkgsubimg = frame.data - bkg
        snr = bkgsubimg/rms
        peaks.append(np.nanmax(bkgsubimg))
        meanbkg.append(np.nanmean(bkg))
        peaksnr.append(np.nanmax(snr))
    peaks = np.array(peaks, dtype=float)
    meanbkg = np.array(meanbkg, dtype=float)
    peaksnr = np.array(peaksnr, dtype=float)

    p["PEAKS"] = peaks
    p["MEAN_BKG"] = meanbkg
    p["PEAK_SNR"] = peaksnr

    p['REF_IMAGE_INDEX'] = np.argmax(peaksnr)
    p['REFERENCE_IMAGE'] = obj_files[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    print(p['REF_IMAGE_INDEX'], "Reference image: {}".format(p['REFERENCE_IMAGE']))

    # get x and y shifts of all images with respect to the first image
    shift_list = compute_shift_list(
        frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=p['REF_IMAGE_INDEX'], skip_failure=True)

    # store shifts in x and y np arrays
    x, y = np.array([]), np.array([])
    for i in range(len(shift_list)):
        x = np.append(x, shift_list[i][0])
        y = np.append(y, shift_list[i][1])
        # print(i,shift_list[i][0],shift_list[i][1])

    # store shifts
    p["XSHIFTS"] = x
    p["YSHIFTS"] = y

    if sort_method == 'MAX_FLUXES':
        sort = np.flip(np.argsort(peaksnr))

    elif sort_method == 'MIN_SHIFTS':
        # calculate median shifts
        median_x = np.nanmedian(x)
        median_y = np.nanmedian(y)
        # calculate relative distances to the median values
        dist = np.sqrt((x - median_x)**2 + (y - median_y)**2)
        # sort in crescent order
        sort = np.argsort(dist)

        if plot:
            indices = np.arange(len(sort))
            plt.plot(indices, dist, 'ko')
            if len(sort) > p['NFILES_FOR_STACK']:
                plt.plot(indices[sort][:p['NFILES_FOR_STACK']],
                         dist[sort][:p['NFILES_FOR_STACK']], 'rx')
            else:
                plt.plot(indices, dist, 'rx')
            plt.xlabel("Image index")
            plt.ylabel("Distance to median position (pixel)")
            plt.show()
    else:
        print("ERROR: sort_method = {} not recognized, select a valid method.".format(
            sort_method))
        exit()    # select files in crescent order based on the distance to the median position

    sorted_files = []
    newsort = []

    # first add reference image
    sorted_files.append(p['REFERENCE_IMAGE'])
    newsort.append(p['REF_IMAGE_INDEX'])

    # then add all valid images to the list of images for stack
    for i in sort:
        # print(obj_files[i],p["XSHIFTS"][i],p["YSHIFTS"][i],p["PEAKS"][i],p["PEAK_SNR"][i],p["MEAN_BKG"][i])
        if np.isfinite(p["XSHIFTS"][i]) and np.isfinite(p["YSHIFTS"][i]) and p["PEAK_SNR"][i] > 1. and i != p['REF_IMAGE_INDEX']:
            sorted_files.append(obj_files[i])
            newsort.append(i)
    newsort = np.array(newsort)

    # Now select up to <N files for stack as defined in the parameters file and save list to the param dict
    if len(sorted_files) > p['NFILES_FOR_STACK']:
        p['SELECTED_FILES_FOR_STACK'] = sorted_files[:p['NFILES_FOR_STACK']]
        p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort[:p['NFILES_FOR_STACK']]
        p['FINAL_NFILES_FOR_STACK'] = p['NFILES_FOR_STACK']
        """
        if correct_shifts :
            # Correct shifts to match the new reference image for the catalog
            p["XSHIFTS"] -= p["XSHIFTS"][newsort[0]]
            p["YSHIFTS"] -= p["YSHIFTS"][newsort[0]]
            """
    else:
        p['FINAL_NFILES_FOR_STACK'] = len(sorted_files)
        p['SELECTED_FILES_FOR_STACK'] = sorted_files
        p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort

    return p


def run_register_frames(p, inframes, inobj_files, info, output_stack="", force=False, maxnsources=0, polarimetry=False):
    """ Pipeline module to register frames

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    inframes : list of frame type len=N_sci_images
        list of frames containing the data of science images
    inobj_files : list of str len=N_sci_images
        list of file paths to the science object image files
    info : dict
        dictionary with additional header cards to include in the header of product
        The following format must be used:
            info = {key1: (value1, comment1), key2: (value2, comment2), ... }
    output_stack : str, optional
        String to define output file name for stacked image
    force : bool, default=False
        whether or not to force reduction if a product already exists
    maxnsources : int, default=0
        maximum number of sources to include in the catalog. Sources are sorted by brightness
        for maxnsources=0 it will include all detected sources
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    frames, obj_files = [], []
    for i in p['SELECTED_FILE_INDICES_FOR_STACK']:
        print(i, inobj_files[i])
        frames.append(deepcopy(inframes[i]))
        obj_files.append(inobj_files[i])

    stack_method = p['SCI_STACK_METHOD']

    # shift_list = compute_shift_list(frames, algorithm='asterism-matching', ref_image=0)
    # print(shift_list)

    # register frames
    registered_frames = register_framedata_list(
        frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=0, inplace=False, skip_failure=True)

    # stack all object files
    combined = imcombine(registered_frames, method=stack_method,
                         sigma_clip=p['SCI_STACK_SIGMA_CLIP'], sigma_cen_func='median', sigma_dev_func='std')

    # get stack data
    img_data = np.array(combined.data, dtype=float)
    err_data = np.array(combined.get_uncertainty())
    mask_data = np.array(combined.mask)

    # get an aperture that's 2 x fwhm measure on the stacked image
    p = calculate_aperture_radius(p, img_data)

    # generate catalog
    p, stack_catalogs = build_catalogs(
        p, img_data, maxnsources=maxnsources, polarimetry=polarimetry, stackmode=True)

    # set master catalogs
    p["CATALOGS"] = stack_catalogs

    # save stack product
    if output_stack != "":
        if not os.path.exists(output_stack) or force:
            # for light products
            s4p.scienceImageLightProduct(obj_files[0], img_data=img_data, info=info, catalogs=p["CATALOGS"], polarimetry=polarimetry,
                                         filename=output_stack, catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=p['WCS_HEADER'], time_key=p["TIME_KEY"])
            # for more complete products with an error and mask extensions
            # s4p.scienceImageProduct(obj_files[0], img_data=img_data, err_data=err_data, mask_data=mask_data, info=info, catalogs=p["CATALOGS"], polarimetry=polarimetry, filename=output_stack, catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=p['WCS_HEADER'], time_key=p["TIME_KEY"])
    return p


def uflux_to_magnitude(uflux):
    """ Pipeline tool to convert uncertain flux into uncertain magnitude

    Parameters
    ----------
    uflux : ufloat
        uncertain flux float value

    Returns
    -------
    umag : ufloat
        uncertain magnitude float value
    """

    try:
        return -2.5 * umath.log10(uflux)
    except:
        return ufloat(np.nan, np.nan)


def flux_to_magnitude(flux):
    """ Pipeline tool to convert flux into magnitude

    Parameters
    ----------
    flux : float
        flux float value

    Returns
    -------
    mag : float
        magnitude float value
    """
    try:
        return -2.5 * np.log10(flux)
    except:
        return np.nan


def calculate_aperture_radius(p, data):
    """ Pipeline tool to calculate aperture radius for photometry

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    data : numpy.ndarray (n x m)
        float array containing the image data

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # calculate background
    bkg, rms = background(data, global_bkg=False)

    # detect sources
    sources = starfind(
        data, threshold=p['PHOT_THRESHOLD'], background=bkg, noise=rms)

    # get fwhm
    fwhm = sources.meta['astropop fwhm']

    p["PHOT_APERTURE_RADIUS"] = p["PHOT_APERTURE_N_X_FWHM"] * fwhm
    p["PHOT_SKYINNER_RADIUS"] = p["PHOT_SKYINNER_N_X_FWHM"] * fwhm
    p["PHOT_SKYOUTER_RADIUS"] = p["PHOT_SKYOUTER_N_X_FWHM"] * fwhm

    return p


def run_aperture_photometry(img_data, x, y, aperture_radius, r_ann, output_mag=True, sortbyflux=True):
    """ Pipeline module to run aperture photometry of sources in an image
    Parameters
    ----------
    img_data : numpy.ndarray (n x m)
        float array containing the image data
    x : list of floats
        list of x coordinates for the sources in the image frame
    y : list of floats
        list of y coordinates for the sources in the image frame
    aperture_radius : float
        aperture radius within which to perform aperture photometry
    r_ann : tuple: (float,float)
        sky annulus inner and outer radii
    output_mag : bool, optional
        to convert output flux into magnitude
    sortbyflux : bool, optional
        to sort output data by flux (brightest first)
    Returns
        x, y, mag, mag_error, smag, smag_error, flags
    -------
     :
    """

    # perform aperture photometry
    ap_phot = aperture_photometry(img_data, x, y, r=aperture_radius, r_ann=r_ann, gain=1.0,
                                  readnoise=None, mask=None, sky_algorithm='mmm')

    # I cannot get round and sharp because they may not match master catalog data
    # ap_phot['round'], ap_phot['sharp'] = sources['round'], sources['sharp']

    # sort table in flux
    if sortbyflux:
        ap_phot.sort('flux')
        # reverse, to start with the highest flux
        ap_phot.reverse()

    x, y = np.array(ap_phot['x']), np.array(ap_phot['y'])
    flux, flux_error = np.array(
        ap_phot['flux']), np.array(ap_phot['flux_error'])
    sky = np.array(ap_phot['sky'])
    # round = np.array(ap_phot['round'])
    # sharp = np.array(ap_phot['sharp'])
    flags = np.array(ap_phot['flags'])

    if output_mag:
        mag, mag_error = np.full_like(flux, np.nan), np.full_like(flux, np.nan)
        smag, smag_error = np.full_like(
            flux, np.nan), np.full_like(flux, np.nan)
        for i in range(len(flux)):
            umag = uflux_to_magnitude(ufloat(flux[i], flux_error[i]))
            mag[i], mag_error[i] = umag.nominal_value, umag.std_dev
            # detnoise = np.sqrt(flux_error[i] * flux_error[i] - flux[i])
            # uskymag = uflux_to_magnitude(ufloat(sky[i],np.sqrt(sky[i]+detnoise*detnoise)))
            uskymag = uflux_to_magnitude(ufloat(sky[i], np.sqrt(sky[i])))
            smag[i], smag_error[i] = uskymag.nominal_value, uskymag.std_dev

        return x, y, mag, mag_error, smag, smag_error, flags

    else:

        return x, y, flux, flux_error, sky, np.sqrt(sky), flags


def read_catalog_coords(catalog):
    """ Pipeline module to read coordinates from a catalog dict
    Parameters
    ----------
    catalog : dict
        catalog of sources containing photometry data
    Returns
        ra : numpy.ndarray (N sources)
            float array containing the right ascension [DEG] data for all sources
        dec : numpy.ndarray (N sources)
            float array containing the declination [DEG] data for all sources
        x : numpy.ndarray (N sources)
            float array containing the x-position [pix] data for all sources
        y : numpy.ndarray (N sources)
            float array containing the y-position [pix] data for all sources
    -------
     :
    """

    # initialize x and y coords of sources
    ra, dec = np.array([]), np.array([])
    x, y = np.array([]), np.array([])

    for star in catalog.keys():
        ra = np.append(ra, catalog[star][1])
        dec = np.append(dec, catalog[star][2])
        x = np.append(x, catalog[star][3])
        y = np.append(y, catalog[star][4])

    return ra, dec, x, y


def set_wcs(p):
    """ Pipeline module to set WCS header parameters from an input header of a
            reference image. The reference image is usually an astrometric field.
    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    Returns
        p with updated WCS keywords
    -------
     :
    """
    ra, dec = p['REF_OBJECT_HEADER']['RA'].split(
        ":"), p['REF_OBJECT_HEADER']['DEC'].split(":")

    ra_str = '{:02d}h{:02d}m{:.2f}s'.format(
        int(ra[0]), int(ra[1]), float(ra[2]))
    dec_str = '{:02d}d{:02d}m{:.2f}s'.format(
        int(dec[0]), int(dec[1]), float(dec[2]))
    # print(ra_str, dec_str)

    coord = SkyCoord(ra_str, dec_str, frame='icrs')
    ra_deg, dec_deg = coord.ra.degree, coord.dec.degree
    p['RA_DEG'], p['DEC_DEG'] = ra_deg, dec_deg
    p['WCS'] = WCS(fits.getheader(p["ASTROM_REF_IMG"], 0), naxis=2)
    p['WCS_HEADER'] = p['WCS'].to_header(relax=True)
    p['WCS_HEADER']['CRVAL1'] = ra_deg
    p['WCS_HEADER']['CRVAL2'] = dec_deg
    # p['WCS_HEADER']['LATPOLE'] = 0
    # p['WCS_HEADER']['LONPOLE'] = 180
    del p['WCS_HEADER']['DATE-OBS']
    del p['WCS_HEADER']['MJD-OBS']

    return p


def generate_catalogs(p, data, sources, fwhm, catalogs=[], catalogs_label='', aperture_radius=10, r_ann=(25, 50), sortbyflux=True, maxnsources=0, polarimetry=False, use_e_beam_for_astrometry=False, solve_astrometry=False):
    """ Pipeline module to generate new catalogs and append it
    to a given list of catalogs
    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    Returns
        catalogs: list of dicts
            returns a list of catalog dictionaries, where the new catalogs
            generatered within this method are appended to an input list
    -------
     :
    """
    current_catalogs_len = len(catalogs)

    if polarimetry:
        catalogs.append({})
        catalogs.append({})

        # no input catalogs, then create new ones
        dx, dy = estimate_dxdy(sources['x'], sources['y'])
        pairs = match_pairs(sources['x'], sources['y'],
                            dx, dy, tolerance=p["MATCH_PAIRS_TOLERANCE"])

        sources_table = Table()

        sources_table['star_index'] = np.arange(len(pairs))+1
        sources_table['x_o'] = sources['x'][pairs['o']]
        sources_table['y_o'] = sources['y'][pairs['o']]
        sources_table['x_e'] = sources['x'][pairs['e']]
        sources_table['y_e'] = sources['y'][pairs['e']]

        # s4plt.plot_sci_polar_frame(data, bkg, sources_table)
        # print("sources:\n",sources)
        # print("\n\nsources_table:\n",sources_table)

        xo, yo, mago, mago_error, smago, smago_error, flagso = run_aperture_photometry(
            data, sources_table['x_o'], sources_table['y_o'], aperture_radius, r_ann, output_mag=True, sortbyflux=False)

        sorted = np.full_like(xo, True, dtype=bool)
        if sortbyflux:
            sorted = np.argsort(mago)

        xe, ye, mage, mage_error, smage, smage_error, flagse = run_aperture_photometry(
            data, sources_table['x_e'], sources_table['y_e'], aperture_radius, r_ann, output_mag=True, sortbyflux=False)

        xo, yo = xo[sorted], yo[sorted]
        mago, mago_error = mago[sorted], mago_error[sorted]
        smago, smago_error = smago[sorted], smago_error[sorted]
        flagso = flagso[sorted]

        xe, ye = xe[sorted], ye[sorted]
        mage, mage_error = mage[sorted], mage_error[sorted]
        smage, smage_error = smage[sorted], smage_error[sorted]
        flagse = flagse[sorted]

        fwhmso, fwhmse = np.full_like(mago, fwhm), np.full_like(mage, fwhm)

        if use_e_beam_for_astrometry:
            xs_for_astrometry = xo
            ys_for_astrometry = yo
        else:
            xs_for_astrometry = xe
            ys_for_astrometry = ye

        if solve_astrometry:
            if use_e_beam_for_astrometry:
                fluxes_for_astrometry = 10**(-0.4*mage)
            else:
                fluxes_for_astrometry = 10**(-0.4*mago)

            h, w = np.shape(data)

            # print ("I'm trying now to solve astrometry ...")
            # print("INPUT parameters: ",xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, h, w, p['REF_OBJECT_HEADER'], {'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE']-0.015, 'scale-units': 'arcsecperpix', 'scale-high':p['PLATE_SCALE']+0.015, 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER']})
            try:
                # Solve astrometry
                # p['WCS'] = solve_astrometry_xy(xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, image_height=h, image_width=w, image_header=p['REF_OBJECT_HEADER'], image_params={'ra': p['RA_DEG'],'dec': p['DEC_DEG'],'pltscl': p['PLATE_SCALE']}, return_wcs=True)
                # solution = solve_astrometry_xy(xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, height=h, width=w, image_header=p['REF_OBJECT_HEADER'], options={'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE']-0.02, 'scale-units': 'arcsecperpix', 'scale-high':p['PLATE_SCALE']+0.02, 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER']})
                solution = solve_astrometry_xy(xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, w, h, options={'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p[
                                               'PLATE_SCALE']-0.02, 'scale-high': p['PLATE_SCALE']+0.02, 'scale-units': 'arcsecperpix', 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER'], 'add_path': p['ASTROM_INDX_PATH']})
                p['WCS'] = solution.wcs
                p['WCS_HEADER'] = p['WCS'].to_header(relax=True)

            except:
                print("WARNING: could not solve astrometry, using WCS from database")

        ras, decs = p['WCS'].all_pix2world(
            xs_for_astrometry, ys_for_astrometry, 0)

        nsources = len(mago)
        if maxnsources:
            nsources = maxnsources

        # save photometry data into the catalogs
        for i in range(nsources):
            catalogs[current_catalogs_len]["{}".format(i)] = (
                i, ras[i], decs[i], xo[i], yo[i], fwhmso[i], fwhmso[i], mago[i], mago_error[i], smago[i], smago_error[i], aperture_radius, flagso[i])
            catalogs[current_catalogs_len+1]["{}".format(i)] = (i, ras[i], decs[i], xe[i], ye[i], fwhmse[i],
                                                                fwhmse[i], mage[i], mage_error[i], smage[i], smage_error[i], aperture_radius, flagse[i])
    else:
        catalogs.append({})

        # x, y = np.array(sources['x']), np.array(sources['y'])
        x, y, mag, mag_error, smag, smag_error, flags = run_aperture_photometry(
            data, sources['x'], sources['y'], aperture_radius, r_ann, output_mag=True, sortbyflux=sortbyflux)

        fwhms = np.full_like(mag, fwhm)

        if solve_astrometry:
            fluxes_for_astrometry = 10**(-0.4*mag)
            h, w = np.shape(data)

            try:
                # image_header=p['REF_OBJECT_HEADER']
                solution = solve_astrometry_xy(x, y, fluxes_for_astrometry, w, h, options={'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE'] -
                                               0.02, 'scale-high': p['PLATE_SCALE']+0.02, 'scale-units': 'arcsecperpix', 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER'], 'add_path': p['ASTROM_INDX_PATH']})
                p['WCS'] = solution.wcs
                p['WCS_HEADER'] = p['WCS'].to_header(relax=True)
            except:
                print("WARNING: could not solve astrometry, using WCS from database")

        ras, decs = p['WCS'].all_pix2world(x, y, 0)

        nsources = len(mag)
        if maxnsources:
            nsources = maxnsources

        # save photometry data into the catalog
        for i in range(nsources):
            catalogs[current_catalogs_len]["{}".format(i)] = (
                i, ras[i], decs[i], x[i], y[i], fwhms[i], fwhms[i], mag[i], mag_error[i], smag[i], smag_error[i], aperture_radius, flags[i])

    return catalogs


def set_sky_aperture(p, aperture_radius):
    """ Pipeline module to calculate the sky aperture radius
    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    aperture_radius : int
        photometry aperture radius in pixels
    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    r_ann = deepcopy(p['PHOT_FIXED_R_ANNULUS'])

    r_in_ann = r_ann[0]

    if r_ann[0] < aperture_radius + p['PHOT_MIN_OFFSET_FOR_SKYINNERRADIUS']:
        r_in_ann = aperture_radius + p['PHOT_MIN_OFFSET_FOR_SKYINNERRADIUS']

    r_out_ann = r_ann[1]
    if r_out_ann < r_in_ann + p['PHOT_MIN_OFFSET_FOR_SKYOUTERRADIUS']:
        r_out_ann = r_in_ann + p['PHOT_MIN_OFFSET_FOR_SKYOUTERRADIUS']

    r_ann = (r_in_ann, r_out_ann)

    return r_ann


def build_catalogs(p, data, catalogs=[], xshift=0., yshift=0., solve_astrometry=True, maxnsources=0, polarimetry=False, stackmode=False):
    """ Pipeline module to generate the catalogs of sources from an image
        This module will perform the following tasks:
        1. Calculate background in the input image
        2. Detect sources using starfind
        3. Perform aperture photometry
        4. Build catalog container and stores it into the parameter dict

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    data : numpy.ndarray (n x m)
        float array containing the image data
    catalogs : list of dicts, optional
        list of dictionaries containing the input catalogs. If not provided it
        will re-detect sources on image and build a new catalog
    xshift : float, optional
        image offset in the x direction with respect to the input catalog
    yshift : float, optional
        image offset in the y direction with respect to the input catalog
    maxnsources : int, default=0
        maximum number of sources to include in the catalog. Sources are sorted by brightness
        for maxnsources=0 it will include all detected sources
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources
    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    catalogs : list of dicts
        output catalogs
    """
    # read image data
    # hdul = fits.open(image_name, mode = "readonly")
    # data = np.array(hdul[0].data, dtype=float)

    # calculate background
    bkg, rms = background(data, global_bkg=False)

    # detect sources
    sources = starfind(
        data, threshold=p["PHOT_THRESHOLD"], background=bkg, noise=rms)

    # get fwhm
    fwhm = sources.meta['astropop fwhm']

    # set aperture radius from
    aperture_radius = p['PHOT_FIXED_APERTURE']
    r_ann = set_sky_aperture(p, aperture_radius)

    p = set_wcs(p)

    # print("******* DEBUG ASTROMETRY **********")
    # print("RA_DEG={} DEC_DEG={}".format(p['RA_DEG'], p['DEC_DEG']))
    # print("WCS:\n{}".format(p['WCS']))

    if stackmode:
        catalogs = []

    if catalogs == []:
        print("Creating new catalog of detected sources:".format(xshift, yshift))

        # print sources table
        # print(sources)

        # print("Running aperture photometry with aperture_radius={} r_ann={}".format(aperture_radius,r_ann))
        catalogs = generate_catalogs(p, data, sources, fwhm, catalogs, aperture_radius=aperture_radius,
                                     r_ann=r_ann, polarimetry=polarimetry, solve_astrometry=True)

        if p['MULTI_APERTURES']:
            for i in range(len(p['PHOT_APERTURES'])):
                aperture_radius = p['PHOT_APERTURES'][i]
                r_ann = set_sky_aperture(p, aperture_radius)
                # print("Running aperture photometry with aperture_radius={} r_ann={}".format(aperture_radius,r_ann))
                catalogs = generate_catalogs(p, data, sources, fwhm, catalogs, aperture_radius=aperture_radius,
                                             r_ann=r_ann, polarimetry=polarimetry, solve_astrometry=False)
    else:

        print("Running aperture photometry for catalogs with an offset by dx={} dy={}".format(
            xshift, yshift))

        for j in range(len(catalogs)):
            # load coordinates from an input catalog
            ras, decs, x, y = read_catalog_coords(catalogs[j])

            # apply shifts
            if np.isfinite(xshift):
                x += xshift
            if np.isfinite(yshift):
                y += yshift

            aperture_radius = catalogs[j]['0'][11]
            r_ann = set_sky_aperture(p, aperture_radius)

            # print("Running aperture photometry for catalog={} xshift={} yshift={} with aperture_radius={} r_ann={}".format(j,xshift,yshift,aperture_radius,r_ann))

            # run aperture photometry
            x, y, mag, mag_error, smag, smag_error, flags = run_aperture_photometry(
                data, x, y, aperture_radius, r_ann, output_mag=True, sortbyflux=False)
            fwhms = np.full_like(mag, fwhm)

            # save data back into the catalog
            for i in range(len(mag)):
                catalogs[j]["{}".format(i)] = (i, ras[i], decs[i], x[i], y[i], fwhms[i], fwhms[i],
                                               mag[i], mag_error[i], smag[i], smag_error[i], aperture_radius, flags[i])

    return p, catalogs


def phot_time_series(sci_list,
                     reduce_dir="./",
                     ts_suffix="",
                     time_key='DATE-OBS',
                     time_format='isot',
                     time_scale='utc',
                     longitude=-45.5825,
                     latitude=-22.5344,
                     altitude=1864,
                     catalog_names=[],
                     time_span_for_rms=5,
                     best_apertures=True,
                     force=True):
    """ Pipeline module to calculate photometry differential time series for a given list of sparc4 sci image products

    Parameters
    ----------
    sci_list : list
        list of paths to science image products
    ts_suffix : str (optional)
        time series suffix to add into the output file name
    reduce_dir : str (optional)
        path to the reduce directory
    time_keyword : str (optional)
        Time keyword in fits header. Default is 'DATE-OBS'
    time_format : str (optional)
        Time format in fits header. Default is 'isot'
    time_scale : str (optional)
        Time scale in fits header. Default is 'utc'
    longitude : float (optional)
        East geographic longitude [deg] of observatory; default is OPD longitude of -45.5825 degrees
    latitude : float (optional)
        North geographic latitude [deg] of observatory; default is OPD latitude of -22.5344 degrees
    altitude : float (optional)
        Observatory elevation [m] above sea level. Default is OPD altitude of 1864 m
    catalog_names : list (optional)
        list of catalog names to be included in the final data product. The less the faster. Default is [], which means all catalogs.
    time_span_for_rms : float, optional
        Time span (in minutes) around a given observation to include other observations to
        calculate running rms.
    best_apertures : bool, optional
        Boolean to include extension with best apertures
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists

    Returns
    -------
    output : str
        path to the output time series product file
    """

    # set output light curve product file name
    output = os.path.join(reduce_dir, "{}_lc.fits".format(ts_suffix))
    if os.path.exists(output) and not force:
        return output

    # initialize data container as dict
    tsdata = {}

    # get information from the first image in the time series
    hdul = fits.open(sci_list[0])
    hdr = hdul[0].header

    if catalog_names == []:
        for hdu in hdul:
            if hdu.name != "PRIMARY":
                catalog_names.append(hdu.name)

    # get number of exposures in time series
    nexps = len(sci_list)

    # bool to get time info only once:
    has_time_info = False

    # apertures array
    apertures = {}

    # loop below to get photometry data for all catalogs
    for key in catalog_names:

        print("Packing time series data for catalog: {}".format(key))

        catdata = s4p.readPhotTimeSeriesData(sci_list,
                                             catalog_key=key,
                                             longitude=longitude,
                                             latitude=latitude,
                                             altitude=altitude,
                                             time_keyword=time_key,
                                             time_format=time_format,
                                             time_scale=time_scale,
                                             time_span_for_rms=5)

        apertures[key] = fits.getheader(sci_list[0], key)['APRADIUS']

        if not has_time_info:
            # get time array
            times = catdata[catdata["SRCINDEX"] == 0]["TIME"]

            # get first and last times
            tstart = Time(times[0], format='jd', scale='utc')
            tstop = Time(times[-1], format='jd', scale='utc')

            # get number of sources
            nsources = fits.getheader(sci_list[0], key)['NOBJCAT']

            has_time_info = True

        tsdata[key] = catdata

        del catdata

    # Construct information dictionary to add to the header of FITS product
    info = {}

    info['OBSERV'] = ('OPD', 'observatory')
    info['OBSLAT'] = (latitude, '[DEG] observatory latitude (N)')
    info['OBSLONG'] = (longitude, '[DEG] observatory longitude (E)')
    info['OBSALT'] = (altitude, '[m] observatory altitude')
    info['TELESCOP'] = ('OPD-PE 1.6m', 'telescope')
    info['INSTRUME'] = ('SPARC4', 'instrument')
    info['OBJECT'] = (hdr["OBJECT"], 'ID of object of interest')
    equinox = 'J2000.0'
    source = SkyCoord(hdr["RA"], hdr["DEC"], unit=(
        u.hourangle, u.deg), frame='icrs', equinox=equinox)
    info['RA'] = (source.ra.value, '[DEG] RA of object of interest')
    info['DEC'] = (source.dec.value, '[DEG] DEC of object of interest')
    info['RADESYS'] = ('ICRS    ', 'reference frame of celestial coordinates')
    info['EQUINOX'] = (2000.0, 'equinox of celestial coordinate system')
    info['PHZEROP'] = (0., '[mag] photometric zero point')
    info['PHOTSYS'] = ("SPARC4", 'photometric system')
    info['TIMEKEY'] = (time_key, 'keyword used to extract times')
    info['TIMEFMT'] = (time_format, 'time format in img files')
    info['TIMESCL'] = (time_scale, 'time scale')
    info['POLAR'] = (False, 'polarimetry mode')
    info['TSTART'] = (tstart.jd, 'observation start time in BJD')
    info['TSTOP'] = (tstop.jd, 'observation stop time in BJD')
    info['DATE-OBS'] = (tstart.isot, 'TSTART as UTC calendar date')
    info['DATE-END'] = (tstop.isot, 'TSTOP as UTC calendar date')
    info['NEXPS'] = (nexps, 'number of exposures')
    info['NSOURCES'] = (nsources, 'number of sources')

    if best_apertures:
        minrms = np.arange(nsources)*0 + 1e20
        selected_apertures = np.zeros_like(minrms)
        selected_data = []
        for i in range(nsources):
            selected_data.append(None)

        # loop over each aperture container
        for key in tsdata.keys():

            # get aperture table data
            tbl = tsdata[key]

            # save minimum rms for each source
            for i in range(nsources):
                loc_tbl = tbl[tbl["SRCINDEX"] == i]
                m_rms = np.nanmedian(loc_tbl['RMS'])
                if m_rms < minrms[i]:
                    minrms[i] = m_rms
                    selected_apertures[i] = apertures[key]
                    loc_tbl["APRADIUS"] = np.full_like(
                        loc_tbl["MAG"], apertures[key])
                    selected_data[i] = loc_tbl

        # add selected data into data container
        tsdata["BEST_APERTURES"] = vstack(selected_data)
        apertures["BEST_APERTURES"] = np.nanmedian(selected_apertures)

    # generate the photometric time series product
    s4p.photTimeSeriesProduct(tsdata, apertures, info=info, filename=output)

    return output


def get_waveplate_angles(wppos):
    """ Pipeline module to get position angles of the wave plate
    given the position index(es)

    Parameters
    ----------
    wppos :
        index for the position angle of the wave plate

    Returns
    -------
    angles : float
        return the angle(s) corresponding to the input position index(es)
    """

    angles = [0., 22.5, 45., 67.5, 90.,
              112.5, 135., 157.5, 180.,
              202.5, 225., 247.5, 270.,
              292.5, 315., 337.5]

    return angles[wppos]


def load_list_of_sci_image_catalogs(sci_list, wppos_key="WPPOS", polarimetry=False):
    """ Pipeline module to load information in the catalog extensions of a list of
    science image products

    Parameters
    ----------
    sci_list : list
        input list of file paths
    wppos_key : str
        image header keyword to get the wave plate position information
    polarimetry : bool
        whether or not in polarimetry mode

    Returns
    -------
    beam or (beam1 and beam2): dict
        dict container to store catalog data
    waveplate_angles : list, only if polarimetry==True
        position angles of the wave plate
    apertures: list
        aperture radii using in the photometry
    nsources: int
        number of sources in the catalog
    """

    nsources = 0

    if polarimetry:
        beam1, beam2 = {}, {}

        waveplate_angles = np.array([])
        apertures = np.array([])

        for i in range(len(sci_list)):

            hdulist = fits.open(sci_list[i])

            wppos = int(hdulist[0].header[wppos_key])
            waveplate_angles = np.append(
                waveplate_angles, get_waveplate_angles(wppos-1))

            phot1data, phot2data = [], []

            for ext in range(1, len(hdulist)):
                if (ext % 2) != 0:
                    if i == 0:
                        apertures = np.append(
                            apertures, hdulist[ext].data[0][11])
                        nsources = len(hdulist[ext].data)

                    phot1data.append(Table(hdulist[ext].data))
                    phot2data.append(Table(hdulist[ext+1].data))
                else:
                    continue
            beam1[sci_list[i]] = phot1data
            beam2[sci_list[i]] = phot2data

        # return beam1, beam2, waveplate_angles*u.degree, apertures, nsources
        return beam1, beam2, waveplate_angles, apertures, nsources

    else:
        beam = {}

        apertures = np.array([])

        for i in range(len(sci_list)):

            hdulist = fits.open(sci_list[i])

            photdata = []

            for ext in range(1, len(hdulist)):
                if i == 0:
                    apertures = np.append(apertures, hdulist[ext].data[0][11])
                    nsources = len(hdulist[ext].data)

                photdata.append(Table(hdulist[ext].data))

            beam[sci_list[i]] = photdata

        return beam, apertures, nsources


def get_qflux(beam, filename, aperture_index, source_index, magkey="MAG", emagkey="EMAG"):
    """ Pipeline module to get catalog fluxes into Qfloat

    Parameters
    ----------
    beam : dict
        dict container to store catalog data
    filename : str
        file name  to select only data for a given exposure
    aperture_index : int
        aperture index to select only data for a given aperture
    source_index : int
        source index to select only data for a given source
    magkey : str
        magnitude keyword
    emagkey : str
        magnitude error keyword

    Returns
    -------
    qflux : Qfloat
        return flux+/-flux_err information from input catalog
    """

    umag = ufloat(beam[filename][aperture_index][magkey][source_index],
                  beam[filename][aperture_index][emagkey][source_index])

    uflux = 10**(-0.4*umag)

    qflux = QFloat(uflux.nominal_value, uflux.std_dev)

    return qflux


def get_photometric_data_for_polar_catalog(beam1, beam2, sci_list, aperture_index=8, source_index=0):
    """ Pipeline module to get photometric data from the catalogs of a
    list of polarimetry exposures

    Parameters
    ----------
    beam1, beam2 : dict
        dict containers to store photometry catalog data
    sci_list : list
        input list of file paths
    aperture_index : int
        aperture index to select only data for a given aperture
    source_index : int
        source index to select only data for a given source

    Returns
    -------
    ra : numpy.ndarray (N sources)
        float array containing the right ascension [DEG] data for all sources
    dec : numpy.ndarray (N sources)
        float array containing the declination [DEG] data for all sources
    x1, y1, x2, y2: numpy.ndarray (N sources)
        float arrays containing the x- and y-positions [pix] data for all sources
    mag.nominal : numpy.ndarray (N sources)
        float array containing the instrumental magnitude data for all sources
    mag.std_dev : numpy.ndarray (N sources)
        float array containing the instrumental magnitude error data for all sources
    fwhm : float
        median full-width at half-maximum (pix)
    skymag.nominal : numpy.ndarray (N sources)
        float array containing the mean instrumental sky magnitude data for all sources
    skymag.std_dev : numpy.ndarray (N sources)
        float array containing the mean instrumental sky magnitude error data for all sources
    flag : int
        sum of photometric flags

    """

    i, j = int(aperture_index), int(source_index)

    ra, dec = np.nan, np.nan
    x1, y1 = np.nan, np.nan
    x2, y2 = np.nan, np.nan

    flux1, flux2 = QFloat(0, 0), QFloat(0, 0)
    skyflux1, skyflux2 = QFloat(0, 0), QFloat(0, 0)

    fwhms = np.array([])

    flag1, flag2 = 0, 0

    for k in range(len(sci_list)):

        if k == 0:
            ra, dec = beam1[sci_list[0]
                            ][i]["RA"][j], beam1[sci_list[0]][i]["DEC"][j]
            x1, y1 = beam1[sci_list[0]
                           ][i]["X"][j], beam1[sci_list[0]][i]["Y"][j]
            x2, y2 = beam2[sci_list[0]
                           ][i]["X"][j], beam2[sci_list[0]][i]["Y"][j]

        flux1 += get_qflux(beam1, sci_list[k], i, j)
        flux2 += get_qflux(beam2, sci_list[k], i, j)

        skyflux1 += get_qflux(beam1, sci_list[k],
                              i, j, magkey="SKYMAG", emagkey="ESKYMAG")
        skyflux2 += get_qflux(beam2, sci_list[k],
                              i, j, magkey="SKYMAG", emagkey="ESKYMAG")

        flag1 += beam1[sci_list[k]][i]["FLAG"][j]
        flag2 += beam2[sci_list[k]][i]["FLAG"][j]

        fwhms = np.append(fwhms, beam1[sci_list[k]][i]["FWHMX"][j])
        fwhms = np.append(fwhms, beam1[sci_list[k]][i]["FWHMY"][j])
        fwhms = np.append(fwhms, beam2[sci_list[k]][i]["FWHMX"][j])
        fwhms = np.append(fwhms, beam2[sci_list[k]][i]["FWHMY"][j])

    fwhm = np.nanmedian(fwhms)

    flux = flux1 + flux2
    mag = -2.5 * np.log10(flux)

    skyflux = (skyflux1 + skyflux2) / (2 * len(sci_list))
    skymag = -2.5 * np.log10(skyflux)

    flag = flag1 + flag2

    return ra, dec, x1, y1, x2, y2, mag.nominal, mag.std_dev, fwhm, skymag.nominal, skymag.std_dev, flag


def compute_polarimetry(sci_list, output_filename="", wppos_key='WPPOS', save_output=True, wave_plate='halfwave', compute_k=True, fit_zero=False, zero=0, base_aperture=8, force=False):
    """ Pipeline module to compute polarimetry for given polarimetric sequence and
        saves the polarimetry data into a FITS SPARC4 product

    Parameters
    ----------
    sci_list : list
        input list of file paths
    output_filename : str (optional)
        file path to save output product, if not given,
         an automatic path convention will be used
    wppos_key : str
        image header keyword to get the wave plate position information
    wave_plate : str
        type of wave plate used in the sequence.
        Accepted values are: "halfwave" or "quarterwave"
    compute_k : bool
        whether or not to include a constant "k" in the polarization model
    zero : float
        zero of polarization in degrees
    fit_zero : bool
        whether or not to fit zero of polarization
    base_aperture : int
        base aperture index
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists

    Returns
    -------
    output_filename : str
        file path for the output product
    """

    # define output polarimetry product file name
    if output_filename == "":
        if wave_plate == 'halfwave':
            output_filename = sci_list[0].replace(
                "_proc.fits", "_l2_polar.fits")
        elif wave_plate == 'quarterwave':
            output_filename = sci_list[0].replace(
                "_proc.fits", "_l4_polar.fits")
        else:
            print("ERROR: wave plate mode not supported, exiting ...")
            exit()

    if os.path.exists(output_filename) and not force:
        print("There is already a polarimetry product :", output_filename)
        return output_filename

    # get data from a list of science image products
    beam1, beam2, waveplate_angles, apertures, nsources = load_list_of_sci_image_catalogs(
        sci_list, wppos_key=wppos_key, polarimetry=True)

    print("Number of sources in catalog: {}".format(nsources))
    print("Number of apertures: {}  varying from {} to {} in steps of {} pix".format(len(
        apertures), apertures[0], apertures[-1], np.abs(np.nanmedian(apertures[1:]-apertures[:-1]))))

    # set number of free parameters equals 2: u and q
    number_of_free_params = 2

    if wave_plate == 'halfwave':
        zero = 0

    # add one parameter: v
    if wave_plate == 'quarterwave':
        number_of_free_params += 1

    # if zero is free, then add another parameter
    if fit_zero:
        zero = None
        # we do not fit zero and k simultaneously
        compute_k = False
        number_of_free_params += 1

    # initialize astropop SLSDualBeamPolarimetry object
    pol = SLSDualBeamPolarimetry(
        wave_plate, compute_k=compute_k, zero=zero, iter_tolerance=1e-6)

    # create an empty dict to store polar catalogs
    polar_catalogs = {}

    # set names and data format for each column in the catalog table
    variables = ['APERINDEX', 'APER', 'SRCINDEX',
                 'RA', 'DEC', 'X1', 'Y1', 'X2', 'Y2',
                 'FWHM', 'MAG', 'EMAG',
                 'SKYMAG', 'ESKYMAG', 'PHOTFLAG',
                 'Q', 'EQ', 'U', 'EU', 'V', 'EV',
                 'P', 'EP', 'THETA', 'ETHETA',
                 'K', 'EK', 'ZERO', 'EZERO', 'NOBS', 'NPAR',
                 'CHI2', 'POLARFLAG']

    for i in range(len(sci_list)):
        variables.append('FO{:04d}'.format(i))
        variables.append('EFO{:04d}'.format(i))
        variables.append('FE{:04d}'.format(i))
        variables.append('EFE{:04d}'.format(i))

    # Initialize container to store polarimetric data
    aperture_keys = []
    for i in range(len(apertures)):
        key = "POLARIMETRY_AP{:03d}".format(int(apertures[i]))
        aperture_keys.append(key)
        polar_catalogs[key] = {}
        for var in variables:
            polar_catalogs[key][var] = np.array([])

    # loop over each aperture
    for i in range(len(apertures)):

        print("Calculating {} polarimetry for aperture {} of {}".format(
            wave_plate, i+1, len(apertures)))

        # loop over each source in the catalog
        for j in range(nsources):

            # retrieve photometric information in a pair of polar catalog
            ra, dec, x1, y1, x2, y2, mag, mag_err, fwhm, skymag, skymag_err, photflag = get_photometric_data_for_polar_catalog(
                beam1, beam2, sci_list, aperture_index=i, source_index=j)

            n_fo, n_fe = [], []
            en_fo, en_fe = [], []

            for filename in sci_list:
                # get fluxes as qfloats
                fo = get_qflux(beam1, filename, i, j)
                fe = get_qflux(beam2, filename, i, j)
                # print(filename, i, j, fo, fe)
                n_fo.append(fo.nominal)
                n_fe.append(fe.nominal)
                en_fo.append(fo.std_dev)
                en_fe.append(fe.std_dev)

            n_fo, n_fe = np.array(n_fo), np.array(n_fe)
            en_fo, en_fe = np.array(en_fo), np.array(en_fe)

            zi, zi_err = np.full_like(n_fo, np.nan), np.full_like(n_fo, np.nan)

            keep = np.isfinite(n_fo)
            keep &= np.isfinite(n_fe)
            keep &= (n_fo > np.nanmedian(en_fo)) & (n_fe > np.nanmedian(en_fe))
            keep &= (en_fo > 0) & (en_fe > 0)

            valid_sci_list = []
            for k in range(len(sci_list)):
                if keep[k]:
                    valid_sci_list.append(sci_list[k])

            number_of_observations = len(valid_sci_list)

            polar_flag = 1
            chi2 = np.nan
            qpol, q_err = np.nan, np.nan
            upol, u_err = np.nan, np.nan
            vpol, v_err = np.nan, np.nan
            ptot, ptot_err = np.nan, np.nan
            theta, theta_err = np.nan, np.nan
            k_factor, k_factor_err = np.nan, np.nan
            zero_err = np.nan

            observed_model = np.full_like(waveplate_angles[keep], np.nan)

            try:
                # compute polarimetry
                norm = pol.compute(
                    waveplate_angles[keep], n_fo[keep], n_fe[keep], f_ord_error=en_fo[keep], f_ext_error=en_fe[keep])

                if wave_plate == 'halfwave':
                    observed_model = halfwave_model(
                        waveplate_angles[keep], norm.q.nominal, norm.u.nominal)

                elif wave_plate == 'quarterwave':
                    observed_model = quarterwave_model(
                        waveplate_angles[keep], norm.q.nominal, norm.u.nominal, norm.v.nominal, zero=norm.zero.nominal)

                zi[keep] = norm.zi.nominal
                zi_err[keep] = norm.zi.std_dev

                chi2 = np.nansum(((norm.zi.nominal - observed_model)/norm.zi.std_dev)
                                 ** 2) / (number_of_observations - number_of_free_params)

                polar_flag = 0

                qpol, q_err = norm.q.nominal, norm.q.std_dev
                upol, u_err = norm.u.nominal, norm.u.std_dev
                if wave_plate == 'quarterwave':
                    vpol, v_err = norm.v.nominal, norm.v.std_dev
                ptot, ptot_err = norm.p.nominal, norm.p.std_dev
                theta, theta_err = norm.theta.nominal, norm.theta.std_dev
                k_factor = norm.k
                zero, zero_err = norm.zero.nominal, norm.zero.std_dev

            except:
                print("WARNING: could not calculate polarimetry for source_index={} and aperture={} pixels".format(
                    j, apertures[i]))

            var_values = [i, apertures[i], j,
                          ra, dec, x1, y1, x2, y2,
                          fwhm, mag, mag_err,
                          skymag, skymag_err, photflag,
                          qpol, q_err,
                          upol, u_err,
                          vpol, v_err,
                          ptot, ptot_err,
                          theta, theta_err,
                          k_factor, k_factor_err,
                          zero, zero_err,
                          number_of_observations, number_of_free_params,
                          chi2, polar_flag]

            for ii in range(len(n_fo)):
                var_values.append(n_fo[ii])
                var_values.append(en_fo[ii])
                var_values.append(n_fe[ii])
                var_values.append(en_fe[ii])

            for ii in range(len(variables)):
                polar_catalogs[aperture_keys[i]][variables[ii]] = np.append(
                    polar_catalogs[aperture_keys[i]][variables[ii]], var_values[ii])

    if save_output:
        info = {}

        hdr_start = fits.getheader(sci_list[0])
        hdr_end = fits.getheader(sci_list[-1])
        if "OBJECT" in hdr_start.keys():
            info['OBJECT'] = (hdr_start["OBJECT"], 'ID of object of interest')
        if "OBSLAT" in hdr_start.keys():
            info['OBSLAT'] = (hdr_start["OBSLAT"],
                              '[DEG] observatory latitude (N)')
        if "OBSLONG" in hdr_start.keys():
            info['OBSLONG'] = (hdr_start["OBSLONG"],
                               '[DEG] observatory longitude (E)')
        if "OBSALT" in hdr_start.keys():
            info['OBSALT'] = (hdr_start["OBSALT"], '[m] observatory altitude')
        info['TELESCOP'] = ('OPD-PE 1.6m', 'telescope')
        if "INSTRUME" in hdr_start.keys():
            info['INSTRUME'] = (hdr_start["INSTRUME"], 'instrument')
        if "EQUINOX" in hdr_start.keys():
            info['EQUINOX'] = (hdr_start["EQUINOX"],
                               'equinox of celestial coordinate system')
        info['PHZEROP'] = (0., '[mag] photometric zero point')
        info['PHOTSYS'] = ("SPARC4", 'photometric system')
        if "CHANNEL" in hdr_start.keys():
            info['CHANNEL'] = (hdr_start["CHANNEL"], 'Instrument channel')
        info['POLTYPE'] = (wave_plate, 'polarimetry type l/2 or l/4')

        tstart = Time(hdr_start["BJD"], format='jd', scale='utc')

        exptime = hdr_end["EXPTIME"]
        if type(exptime) == str:
            if ',' in exptime:
                exptime = exptime.replace(",", ".")
            exptime = float(exptime)

        tstop = Time(hdr_end["BJD"]+exptime/(24*60*60),
                     format='jd', scale='utc')

        info['TSTART'] = (tstart.jd, 'observation start time in BJD')
        info['TSTOP'] = (tstop.jd, 'observation stop time in BJD')
        info['DATE-OBS'] = (tstart.isot, 'TSTART as UTC calendar date')
        info['DATE-END'] = (tstop.isot, 'TSTOP as UTC calendar date')
        info['NSOURCES'] = (nsources, 'number of sources')
        info['NEXPS'] = (len(sci_list), 'number of exposures in sequence')

        for k in range(len(sci_list)):
            hdr = fits.getheader(sci_list[k])
            info["FILE{:04d}".format(k)] = (
                os.path.basename(sci_list[k]), 'file name of exposure')
            info["EXPT{:04d}".format(k)] = (exptime, 'exposure time (s)')
            info["BJD{:04d}".format(k)] = (
                hdr["BJD"], 'start time of exposure (BJD)')
            info["WPPO{:04d}".format(k)] = (
                hdr[wppos_key], 'WP index position of exposure')
            info["WANG{:04d}".format(k)] = (
                waveplate_angles[k], 'WP angle of exposure (deg)')

        print("Saving output {} polarimetry product: {}".format(
            wave_plate, output_filename))
        output_hdul = s4p.polarProduct(
            polar_catalogs, info=info, filename=output_filename)

    return output_filename


def get_polarimetry_results(filename, source_index=0, aperture_radius=None, min_aperture=0, max_aperture=1024, plot=False, verbose=False):
    """ Pipeline module to compute polarimetry for given polarimetric sequence and
        saves the polarimetry data into a FITS SPARC4 product

    Parameters
    ----------
    filename : str
        file path for a polarimetry product
    source_index : int
        source index to select only data for a given source
    aperture_radius : float
        to select a given aperture radius, if not provided
        then the minimum chi-square solution will be adopted
    min_aperture : float
        minimum aperture radius (pix)
    max_aperture : float
        minimum aperture radius (pix)
    plot: bool
        whether or not to plot results

    Returns
    -------
    loc : dict
        container to store polarimetry results for given target and aperture
    """

    loc = {}
    loc["POLAR_PRODUCT"] = filename
    loc["SOURCE_INDEX"] = source_index

    # open polarimetry product FITS file
    hdul = fits.open(filename)
    wave_plate = hdul[0].header['POLTYPE']

    # initialize aperture index
    aperture_index = 1

    # if an aperture index is not given, then consider the one with minimum chi2
    if aperture_radius == None:
        minchi2 = 1e30
        for i in range(len(hdul)):
            if hdul[i].name == 'PRIMARY' or hdul[i].header['APRADIUS'] < min_aperture or hdul[i].header['APRADIUS'] > max_aperture:
                continue
            curr_chi2 = hdul[i].data[hdul[i].data['SRCINDEX']
                                     == source_index]['CHI2'][0]
            if curr_chi2 < minchi2:
                minchi2 = curr_chi2
                aperture_index = i
    else:
        minapdiff = 1e30
        for i in range(len(hdul)):
            if hdul[i].name == 'PRIMARY':
                continue
            curr_apdiff = np.abs(hdul[i].header['APRADIUS'] - aperture_radius)
            if curr_apdiff < minapdiff:
                minapdiff = curr_apdiff
                aperture_index = i

    # get selected aperture radius
    aperture_radius = hdul[aperture_index].header['APRADIUS']
    loc["APERTURE_INDEX"] = aperture_index
    loc["APERTURE_RADIUS"] = aperture_radius
    # isolate data for selected aperture and target
    tbl = hdul[aperture_index].data[hdul[aperture_index].data['SRCINDEX'] == source_index]

    # get number of exposure in the polarimetric sequence
    nexps = hdul[0].header['NEXPS']
    loc["NEXPS"] = nexps

    # get source magnitude
    mag = QFloat(float(tbl["MAG"][0]), float(tbl["EMAG"][0]))
    loc["MAG"] = mag

    # get source coordinates
    ra, dec = tbl["RA"][0], tbl["DEC"][0]

    loc["RA"] = ra
    loc["DEC"] = dec
    loc["FWHM"] = tbl["FWHM"][0]
    loc["X1"] = tbl["X1"][0]
    loc["Y1"] = tbl["Y1"][0]
    loc["X2"] = tbl["X2"][0]
    loc["Y2"] = tbl["Y2"][0]

    # get polarization data and the WP position angles
    fos, efos = np.arange(nexps)*np.nan, np.arange(nexps)*np.nan
    fes, efes = np.arange(nexps)*np.nan, np.arange(nexps)*np.nan
    zis, zierrs = np.arange(nexps)*np.nan, np.arange(nexps)*np.nan
    waveplate_angles = np.arange(nexps)*np.nan

    for ii in range(nexps):
        fos[ii] = tbl["FO{:04d}".format(ii)]
        efos[ii] = tbl["EFO{:04d}".format(ii)]
        fes[ii] = tbl["FE{:04d}".format(ii)]
        efes[ii] = tbl["EFE{:04d}".format(ii)]
        waveplate_angles[ii] = hdul[0].header["WANG{:04d}".format(ii)]

    # filter out nan data
    keep = (np.isfinite(fos)) & (np.isfinite(fes))
    keep &= (np.isfinite(efos)) & (np.isfinite(efes))

    if len(fos[keep]) == 0 or len(fes[keep]) == 0:
        print("WARNING: no useful polarization data for Source index: {}  and aperture: {} pix ".format(
            source_index, aperture_radius))
        # get polarimetry results
        qpol = QFloat(np.nan, np.nan)
        upol = QFloat(np.nan, np.nan)
        vpol = QFloat(np.nan, np.nan)
        ppol = QFloat(np.nan, np.nan)
        theta = QFloat(np.nan, np.nan)
        kcte = QFloat(np.nan, np.nan)
        zero = QFloat(np.nan, np.nan)
        # cast zi data into QFloat
        fo = QFloat(np.array([np.nan]), np.array([np.nan]))
        fe = QFloat(np.array([np.nan]), np.array([np.nan]))
        zi = QFloat(np.array([np.nan]), np.array([np.nan]))
        n, m = 0, 0
        sig_res = np.array([np.nan])
        chi2 = np.nan
        observed_model = zi
    else:
        # get polarimetry results
        qpol = QFloat(tbl['Q'][0], tbl['EQ'][0])
        upol = QFloat(tbl['U'][0], tbl['EU'][0])
        vpol = QFloat(tbl['V'][0], tbl['EV'][0])
        ppol = QFloat(tbl['P'][0], tbl['EP'][0])
        theta = QFloat(tbl['THETA'][0], tbl['ETHETA'][0])
        kcte = QFloat(tbl['K'][0], tbl['EK'][0])
        zero = QFloat(tbl['ZERO'][0], tbl['EZERO'][0])

        # cast zi data into QFloat
        fo = QFloat(fos[keep], efos[keep])
        fe = QFloat(fes[keep], efes[keep])

        # calculate polarimetry model and get statistical quantities
        observed_model = np.full_like(waveplate_angles[keep], np.nan)
        if wave_plate == "halfwave":
            # initialize astropop SLSDualBeamPolarimetry object
            pol = SLSDualBeamPolarimetry(wave_plate, compute_k=True, zero=0)
            observed_model = halfwave_model(
                waveplate_angles[keep], qpol.nominal, upol.nominal)

        elif wave_plate == "quarterwave":
            # initialize astropop SLSDualBeamPolarimetry object
            pol = SLSDualBeamPolarimetry(
                wave_plate, compute_k=False, zero=zero.nominal)
            observed_model = quarterwave_model(
                waveplate_angles[keep], qpol.nominal, upol.nominal, vpol.nominal, zero=zero.nominal)

        # compute polarimetry
        norm = pol.compute(
            waveplate_angles[keep], fos[keep], fes[keep], f_ord_error=efos[keep], f_ext_error=efes[keep])

        zis[keep] = norm.zi.nominal
        zierrs[keep] = norm.zi.std_dev

        # cast zi data into QFloat
        zi = QFloat(zis[keep], zierrs[keep])

        n, m = tbl['NOBS'][0], tbl['NPAR'][0]
        resids = zi.nominal - observed_model
        sig_res = np.nanstd(resids)
        chi2 = np.nansum((resids/zi.std_dev)**2) / (n - m)

    # print(waveplate_angles[keep], zi, qpol, upol, ppol, theta, kcte)

    # print results
    if verbose:
        print("Source index: i={} ".format(source_index))
        print("Source RA={} Dec={} mag={}".format(ra, dec, mag))
        print("Best aperture radius: {} pixels".format(aperture_radius))
        print("Polarization in Q: {}".format(qpol))
        print("Polarization in U: {}".format(upol))
        print("Polarization in V: {}".format(vpol))
        print("Total linear polarization p: {}".format(ppol))
        print("Angle of polarization theta: {}".format(theta))
        print("Free constant k: {}".format(kcte))
        print("Zero of polarization: {}".format(zero))
        print("RMS of {} residuals: {:.5f}".format("zi", sig_res))
        print("Reduced chi-square (n={}, DOF={}): {:.2f}".format(n, n-m, chi2))

    loc["WAVEPLATE_ANGLES"] = waveplate_angles[keep]
    loc["ZI"] = zi
    loc["FO"] = fo
    loc["FE"] = fe
    loc["OBSERVED_MODEL"] = observed_model
    loc["Q"] = qpol
    loc["U"] = upol
    loc["V"] = vpol
    loc["P"] = ppol
    loc["THETA"] = theta
    loc["K"] = kcte
    loc["ZERO"] = zero
    loc["CHI2"] = chi2
    loc["RMS"] = sig_res
    loc["NOBS"] = n
    loc["NPAR"] = m

    # plot polarization data and best-fit model
    if plot:
        # set title to appear in the plot header
        title_label = r"Source index: {}    aperture: {} pix    $\chi^2$: {:.2f}    RMS: {:.4f}".format(
            source_index, aperture_radius, chi2, sig_res)

        s4plt.plot_polarimetry_results(
            loc, title_label=title_label, wave_plate=wave_plate)

    hdul.close()

    return loc


def psf_analysis(filename, aperture=10, half_windowsize=15, nsources=0, percentile=99.5, polarimetry=False, plot=False, verbose=False):
    """ Pipeline module to compute point spread function analysis and
        save the PSF data into a FITS SPARC4 product

    Parameters
    ----------
    filename : str
        file path for a science image product
    plot: bool
        whether or not to plot results

    Returns
    -------
    loc : dict
        container to store polarimetry results for given target and aperture
    """

    loc = {}
    loc["PSF_PRODUCT"] = filename

    # open science image product FITS file
    hdulist = fits.open(filename)
    img_data = hdulist[0].data[0]
    indices = np.indices(img_data.shape)

    # get pixel scale from the WCS and convert it to arcsec
    wcs_obj = WCS(hdulist[0].header, naxis=2)
    pixel_scale = proj_plane_pixel_scales(wcs_obj)
    pixel_scale *= 3600
    print("Pixel scale: x: {:.3f} arcsec/pix y: {:.3f} arcsec/pix".format(
        pixel_scale[0], pixel_scale[1]))

    if polarimetry:

        catN_label = "CATALOG_POL_N_AP{:03d}".format(aperture)
        catS_label = "CATALOG_POL_S_AP{:03d}".format(aperture)
        photN_data = Table(hdulist[catN_label].data)
        photS_data = Table(hdulist[catS_label].data)

        nsources = len(hdulist[catN_label].data)

        fwhms = np.array([])

        for j in range(nsources):
            xN, yN = photN_data["X"][j], photN_data["Y"][j]
            xS, yS = photS_data["X"][j], photS_data["Y"][j]

            fwhms = np.append(fwhms, photN_data["FWHMX"][j])
            fwhms = np.append(fwhms, photN_data["FWHMY"][j])
            fwhms = np.append(fwhms, photS_data["FWHMX"][j])
            fwhms = np.append(fwhms, photS_data["FWHMY"][j])

    else:

        cat_label = "CATALOG_PHOT_AP{:03d}".format(aperture)
        phot_data = Table(hdulist[cat_label].data)

        if nsources == 0 or nsources > len(hdulist[cat_label].data):
            nsources = len(hdulist[cat_label].data)

        fwhms = np.array([])

        boxes = np.zeros([nsources, 2*half_windowsize+1,
                         2*half_windowsize+1]) * np.nan
        xbox, ybox = None, None
        for j in range(nsources):
            x, y = phot_data["X"][j], phot_data["Y"][j]
            fwhms = np.append(fwhms, phot_data["FWHMX"][j])
            fwhms = np.append(fwhms, phot_data["FWHMY"][j])

            # get  box around source
            box = trim_array(img_data, 2*half_windowsize,
                             (x, y), indices=indices)
            zbox = box[0]

            nx, ny = np.shape(zbox)

            if nx == 2*half_windowsize+1 and ny == 2*half_windowsize+1:
                maxvalue = np.nanmax(zbox)
                nzbox = zbox/maxvalue
                # print(j, maxvalue, np.nanmedian(zbox))
                boxes[j, :, :] = nzbox
                xbox, ybox = box[1]-x, box[2]-y

                # vmin, vmax = np.percentile(nzbox, 1.), np.percentile(nzbox, 99.)
                # s4plt.plot_2d(xbox, ybox, nzbox, LIM=None, LAB=["x (pix)", "y (pix)","flux fraction"], z_lim=[vmin,vmax], title="source: {}".format(j), pfilename="", cmap="gist_heat")

        master_box = np.nanmedian(boxes, axis=0)
        master_box_err = np.nanmedian(np.abs(boxes - master_box), axis=0)

        min = np.percentile(master_box, 0.5)
        master_box = master_box - min

        max = np.nanmax(master_box)
        master_box /= max
        master_box_err /= max

        # vmin, vmax = np.percentile(master_box, 3.), np.percentile(master_box, 97.)
        # s4plt.plot_2d(xbox*0.33, ybox*0.33, master_box, LIM=None, LAB=["x (arcsec)", "y (arcsec)","flux fraction"], z_lim=[vmin,vmax], title="PSF", pfilename="", cmap="gist_heat")

        master_fwhm = _fwhm_loop('gaussian', master_box, xbox, ybox, 0, 0)

        print("Median FWHM: {:.3f} pix   Master PSF FWHM: {:.3f} pix".format(
            np.median(fwhms), master_fwhm))

        # multiply x and y by the pixel scale
        xbox *= pixel_scale[0]
        ybox *= pixel_scale[1]

        # plot PSF results
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False, gridspec_kw={
                                 'hspace': 0.5, 'height_ratios': [1, 1]})

        # plot PSF data
        vmin, vmax = np.percentile(
            master_box, 10), np.percentile(master_box, 99)
        axes[0, 0].pcolor(xbox, ybox, master_box, vmin=vmin,
                          vmax=vmax, shading='auto', cmap="cividis")
        axes[0, 0].plot(xbox[half_windowsize], np.zeros_like(
            xbox[half_windowsize]), '--', lw=1., color='white', zorder=3)
        axes[0, 0].plot(np.zeros_like(ybox[:, half_windowsize]),
                        ybox[:, half_windowsize], '--', lw=1., color='white', zorder=3)
        axes[0, 0].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[0, 0].set_ylabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[0, 0].set_title("PSF data", pad=10, fontsize=20)

        # contour plot
        axes[0, 1].contour(xbox, ybox, master_box,
                           vmin=vmin, vmax=vmax, colors='k')
        axes[0, 1].plot(xbox[half_windowsize], np.zeros_like(
            xbox[half_windowsize]), '--', lw=1., color='k', zorder=3)
        axes[0, 1].plot(np.zeros_like(ybox[:, half_windowsize]),
                        ybox[:, half_windowsize], '--', lw=1., color='k', zorder=3)
        axes[0, 1].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[0, 1].set_ylabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[0, 1].set_title("PSF contour", pad=10, fontsize=20)

        # Fit the data using a Gaussian
        model = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
        fitter = fitting.LevMarLSQFitter()
        best_fit = fitter(model, xbox[half_windowsize], master_box[half_windowsize],
                          weights=1.0/master_box_err[half_windowsize]**2)
        # print(best_fit)

        # plot x profile
        axes[1, 0].set_title("E-W profile \nFWHM: {:.2f} arcsec".format(
            2.355*best_fit.stddev.value), pad=10, fontsize=20)
        axes[1, 0].errorbar(xbox[half_windowsize], master_box[half_windowsize],
                            yerr=master_box_err[half_windowsize], lw=0.5, fmt='o', color='k')
        axes[1, 0].plot(xbox[half_windowsize], best_fit(
            xbox[half_windowsize]), '-', lw=2, color='brown')
        axes[1, 0].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[1, 0].set_ylabel("flux", fontsize=16)
        # axes[1,0].legend(fontsize=16)

        best_fit = fitter(model, ybox[:, half_windowsize], master_box[:,
                          half_windowsize], weights=1.0/master_box_err[:, half_windowsize]**2)
        # print(best_fit)
        # plot y profile
        axes[1, 1].set_title("N-S profile \nFWHM: {:.2f} arcsec".format(
            2.355*best_fit.stddev.value), pad=10, fontsize=20)
        axes[1, 1].errorbar(ybox[:, half_windowsize], master_box[:, half_windowsize],
                            master_box_err[:, half_windowsize], lw=0.5, fmt='o', color='k')
        axes[1, 1].plot(ybox[:, half_windowsize], best_fit(
            ybox[:, half_windowsize]), '-', lw=2, color='darkblue')
        axes[1, 1].set_xlabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[1, 1].set_ylabel("flux", fontsize=16)
        # axes[1,1].legend(fontsize=16)

        plt.show()

        # vmin, vmax = np.percentile(img, 3.), np.percentile(img, 97)
        # s4plt.plot_2d(xw, yw, img, LIM=None, LAB=["x (pix)", "y (pix)","flux fraction"], z_lim=[vmin,vmax], title="PSF plot", pfilename="", cmap="gist_heat")

    return loc


def stack_and_reduce_sci_images(p, sci_list, reduce_dir, ref_img="", stack_suffix="", force=True, match_frames=True, polarimetry=False, verbose=False, plot=False):
    """ Pipeline module to run stack and reduction of science images

    Parameters
    ----------
    p : dict
        params
    sci_list : list
        list of science file paths
    reduce_dir : str
        path to reduce directory
    ref_img : str
        path to a reference image. If empty will take it from "p"
    stack_suffix : str
        suffix to be appended to the output stack file name
    force : bool
        force reduction, even if products already exist
    match_frames : bool
        match frames in sci list
    polarimetry : bool
        is it polarimetry data?
    verbose : bool
        print verbose messages
    plot : bool
        do plots

    Returns
    -------
    p : dict
        params
    """

    # clean list of reduced images from previous channel/object
    if 'OBJECT_REDUCED_IMAGES' in p.keys():
        del p['OBJECT_REDUCED_IMAGES']

    # calculate stack
    p = stack_science_images(p,
                             sci_list,
                             reduce_dir=reduce_dir,
                             force=force,
                             stack_suffix=stack_suffix,
                             polarimetry=polarimetry)

    # set numbe of science reduction loops to avoid memory issues.
    nloops = int(
        np.ceil(len(sci_list) / p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP']))

    # set reference image
    if ref_img == "":
        ref_img = p['REFERENCE_IMAGE']

    for loop in range(nloops):
        first = p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] * loop
        last = p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] * (loop+1)
        if last > len(sci_list):
            last = len(sci_list)

        if verbose:
            print(
                "Running loop {} of {} -> images in loop: {} to {} ... ".format(loop, nloops, first, last))

        # reduce science data and calculate stack
        p = reduce_science_images(p,
                                  sci_list[first:last],
                                  reduce_dir=reduce_dir,
                                  ref_img=ref_img,
                                  force=force,
                                  match_frames=match_frames,
                                  polarimetry=polarimetry)

    # reduce science data and calculate stack
    # p = old_reduce_science_images(p, p['objsInPolarL2data'][j][object], data_dir=data_dir, reduce_dir=reduce_dir, force=options.force, match_frames=match_frames, stack_suffix=stack_suffix, polarimetry=True)
    if match_frames and plot:
        if polarimetry:
            s4plt.plot_sci_polar_frame(p['OBJECT_STACK'], percentile=99.5)
        else:
            s4plt.plot_sci_frame(
                p['OBJECT_STACK'], nstars=20, use_sky_coords=True)

    return p


def polar_time_series(sci_pol_list,
                      reduce_dir="./",
                      ts_suffix="",
                      aperture_radius=None,
                      min_aperture=0,
                      max_aperture=1024,
                      force=True):
    """ Pipeline module to calculate photometry differential time series for a given list of sparc4 sci image products

    Parameters
    ----------
    sci_pol_list : list
        list of paths to science polar products
    ts_suffix : str (optional)
        time series suffix to add into the output file name
    reduce_dir : str (optional)
        path to the reduce directory
    aperture_radius : float (optional)
        select aperture radius. Default is None and it will calculate best aperture
    min_aperture : float
        minimum aperture radius (pix)
    max_aperture : float
        minimum aperture radius (pix)
    force : bool
        force reduction even if product already exists

    Returns
    -------
    output : str
        path to the output time series product file
    """

    # set output light curve product file name
    output = os.path.join(reduce_dir, "{}_ts.fits".format(ts_suffix))
    if os.path.exists(output) and not force:
        return output

    # get information from the first image in the time series
    basehdul = fits.open(sci_pol_list[0])
    basehdr = basehdul[0].header

    # set number of input polar files (size of time series)
    npolfiles = len(sci_pol_list)

    # get number of sources in polar catalog of first sequence:
    nsources = basehdr['NSOURCES']

    # initialize data container as dict
    tsdata = {}

    tsdata['TIME'] = np.array([])
    tsdata['SRCINDEX'] = np.array([])
    tsdata['RA'] = np.array([])
    tsdata['DEC'] = np.array([])
    tsdata['X1'] = np.array([])
    tsdata['Y1'] = np.array([])
    tsdata['X2'] = np.array([])
    tsdata['Y2'] = np.array([])
    tsdata['FWHM'] = np.array([])
    tsdata['MAG'] = np.array([])
    tsdata['EMAG'] = np.array([])

    tsdata['Q'] = np.array([])
    tsdata['EQ'] = np.array([])
    tsdata['U'] = np.array([])
    tsdata['EU'] = np.array([])
    tsdata['V'] = np.array([])
    tsdata['EV'] = np.array([])
    tsdata['P'] = np.array([])
    tsdata['EP'] = np.array([])
    tsdata['THETA'] = np.array([])
    tsdata['ETHETA'] = np.array([])
    tsdata['K'] = np.array([])
    tsdata['EK'] = np.array([])
    tsdata['ZERO'] = np.array([])
    tsdata['EZERO'] = np.array([])
    tsdata['NOBS'] = np.array([])
    tsdata['NPAR'] = np.array([])
    tsdata['CHI2'] = np.array([])

    ti, tf = 0, 0

    for i in range(len(sci_pol_list)):

        print("Packing time series data for polar file {} of {}".format(
            i+1, len(sci_pol_list)))

        hdul = fits.open(sci_pol_list[i])
        header = hdul[0].header

        mid_bjd = (header["TSTART"] + header["TSTOP"]) / 2

        if i == 0:
            ti = header["TSTART"]
        if i == len(sci_pol_list) - 1:
            tf = header["TSTOP"]

        for j in range(nsources):

            # read polarimetry results for the base sequence in the time series
            polar = get_polarimetry_results(sci_pol_list[i],
                                            source_index=j,
                                            aperture_radius=aperture_radius,
                                            min_aperture=min_aperture,
                                            max_aperture=max_aperture)

            tsdata['TIME'] = np.append(tsdata['TIME'], mid_bjd)
            tsdata['SRCINDEX'] = np.append(tsdata['SRCINDEX'], j)

            tsdata['RA'] = np.append(tsdata['RA'], polar['RA'])
            tsdata['DEC'] = np.append(tsdata['DEC'], polar['DEC'])
            tsdata['MAG'] = np.append(tsdata['MAG'], polar['MAG'].nominal)
            tsdata['EMAG'] = np.append(tsdata['EMAG'], polar['MAG'].std_dev)
            tsdata['FWHM'] = np.append(tsdata['FWHM'], polar['FWHM'])
            tsdata['X1'] = np.append(tsdata['X1'], polar['X1'])
            tsdata['Y1'] = np.append(tsdata['Y1'], polar['Y1'])
            tsdata['X2'] = np.append(tsdata['X2'], polar['X2'])
            tsdata['Y2'] = np.append(tsdata['Y2'], polar['Y2'])

            tsdata['Q'] = np.append(tsdata['Q'], polar['Q'].nominal)
            tsdata['EQ'] = np.append(tsdata['EQ'], polar['Q'].std_dev)
            tsdata['U'] = np.append(tsdata['U'], polar['U'].nominal)
            tsdata['EU'] = np.append(tsdata['EU'], polar['U'].std_dev)
            tsdata['V'] = np.append(tsdata['V'], polar['V'].nominal)
            tsdata['EV'] = np.append(tsdata['EV'], polar['V'].std_dev)
            tsdata['P'] = np.append(tsdata['P'], polar['P'].nominal)
            tsdata['EP'] = np.append(tsdata['EP'], polar['P'].std_dev)
            tsdata['THETA'] = np.append(
                tsdata['THETA'], polar['THETA'].nominal)
            tsdata['ETHETA'] = np.append(
                tsdata['ETHETA'], polar['THETA'].std_dev)
            tsdata['K'] = np.append(tsdata['K'], polar['K'].nominal)
            tsdata['EK'] = np.append(tsdata['EK'], polar['K'].std_dev)
            tsdata['ZERO'] = np.append(tsdata['ZERO'], polar['ZERO'].nominal)
            tsdata['EZERO'] = np.append(tsdata['EZERO'], polar['ZERO'].std_dev)
            tsdata['NOBS'] = np.append(tsdata['NOBS'], polar['NOBS'])
            tsdata['NPAR'] = np.append(tsdata['NPAR'], polar['NPAR'])
            tsdata['CHI2'] = np.append(tsdata['CHI2'], polar['CHI2'])

        hdul.close()
        del hdul

    # Construct information dictionary to add to the header of FITS product
    info = {}

    info['OBSERV'] = ('OPD', 'observatory')
    info['OBSLAT'] = (basehdr["OBSLAT"], '[DEG] observatory latitude (N)')
    info['OBSLONG'] = (basehdr["OBSLONG"], '[DEG] observatory longitude (E)')
    info['OBSALT'] = (basehdr["OBSALT"], '[m] observatory altitude')
    info['TELESCOP'] = ('OPD-PE 1.6m', 'telescope')
    info['INSTRUME'] = ('SPARC4', 'instrument')
    info['OBJECT'] = (basehdr["OBJECT"], 'ID of object of interest')
    info['CHANNEL'] = (basehdr["CHANNEL"], 'Instrument channel')
    info['PHZEROP'] = (0., '[mag] photometric zero point')
    info['PHOTSYS'] = ("SPARC4", 'photometric system')
    info['POLTYPE'] = (basehdr["POLTYPE"], 'polarimetry type l/2 or l/4')
    info['NSOURCES'] = (basehdr["NSOURCES"], 'number of sources')

    # get first and last times
    tstart = Time(ti, format='jd', scale='utc')
    tstop = Time(tf, format='jd', scale='utc')
    info['TSTART'] = (tstart.jd, 'observation start time in BJD')
    info['TSTOP'] = (tstop.jd, 'observation stop time in BJD')
    info['DATE-OBS'] = (tstart.isot, 'TSTART as UTC calendar date')
    info['DATE-END'] = (tstop.isot, 'TSTOP as UTC calendar date')

    # generate the photometric time series product
    s4p.polarTimeSeriesProduct(tsdata, info=info, filename=output)

    return output


def select_best_phot_aperture_per_target(filename, plot=False):
    """ Pipeline module to get the best apertures per target

    Parameters
    ----------
    filename : str
        input light curves product (fits)
    plot : bool
        plots

    Returns
    -------
    output : numpy array of floats
        array of best aperture radii for all sources
    """

    # open time series fits file
    hdul = fits.open(filename)

    nsources = hdul[0].header['NSOURCES']

    minrms = np.arange(nsources)*0 + 1e20
    selected_apertures = np.zeros_like(minrms)
    mags, emags = np.full_like(minrms, np.nan),  np.full_like(minrms, np.nan)

    # loop over each hdu
    for hdu in hdul:
        # skip primary hdu
        if hdu.name == "PRIMARY":
            continue

        # get aperture table  data
        tbl = hdul[hdu.name].data

        # get aperture radius value
        aperture_radius = hdu.header["APRADIUS"]

        for i in range(nsources):
            loc_tbl = tbl[tbl["SRCINDEX"] == i]
            m_rms = np.nanmedian(loc_tbl['RMS'])
            if m_rms < minrms[i]:
                minrms[i] = m_rms
                selected_apertures[i] = aperture_radius
                mags[i] = np.nanmedian(loc_tbl['MAG'])
                emags[i] = np.nanmedian(loc_tbl['EMAG'])
            if plot:
                plt.plot([aperture_radius], [m_rms], '.')

    if plot:
        for i in range(nsources):
            plt.plot([selected_apertures[i]], [minrms[i]], 'rx')

        plt.ylabel("RMS [mag]")
        plt.xlabel("Aperture radius [pix]")
        plt.show()

        plt.errorbar(selected_apertures, mags, yerr=emags, fmt='o')
        plt.ylabel("Instrumental magnitude")
        plt.xlabel("Aperture radius [pix]")
        plt.show()

    return selected_apertures
