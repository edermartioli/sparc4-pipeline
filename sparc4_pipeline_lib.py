# -*- coding: iso-8859-1 -*-
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

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys

import sparc4_products as s4p
import sparc4_product_plots as s4plt
import sparc4_params
import sparc4_utils as s4utils

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
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


def init_s4_p(datadir="", reducedir="", nightdir="", channels="", print_report=False) :
    """ Pipeline module to initialize SPARC4 parameters
    Parameters
    ----------
    datadir : str, optional
        String to define the directory path to the raw data
    reducedir : str, optional
        String to define the directory path to the reduced data
    nightdir : str, optional
        String to define the night directory name
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
    p = sparc4_params.load_sparc4_parameters()

    if datadir != "" :
        p['ROOTDATADIR'] = datadir
    
    if reducedir != "" :
        p['ROOTREDUCEDIR'] = reducedir
    
    p['SELECTED_CHANNELS'] = p['CHANNELS']
    if channels != "" :
        p['SELECTED_CHANNELS']  = []
        chs = channels.split(",")
        for ch in chs :
            p['SELECTED_CHANNELS'].append(int(ch))
            
    # if reduced dir doesn't exist create one
    if not os.path.exists(p['ROOTREDUCEDIR']) :
        os.mkdir(p['ROOTREDUCEDIR'])

    #organize files to be reduced
    p = s4utils.identify_files(p, nightdir, print_report=print_report)

    p['ch_reduce_directories'] = []
    p['reduce_directories'] = []
    
    for j in range(len(p['CHANNELS'])) :
        data_dir = p['data_directories'][j]
        ch_reduce_dir = '{}/sparc4acs{}/'.format(p['ROOTREDUCEDIR'],p['CHANNELS'][j])
        reduce_dir = '{}/{}/'.format(ch_reduce_dir,nightdir)
        
        p['ch_reduce_directories'].append(ch_reduce_dir)
        p['reduce_directories'].append(reduce_dir)

        if not os.path.exists(ch_reduce_dir) :
            os.mkdir(ch_reduce_dir)

        # if reduced dir doesn't exist create one
        if not os.path.exists(reduce_dir) :
            os.mkdir(reduce_dir)

    return p



def run_master_calibration(p, inputlist=[], output="", obstype='bias', data_dir="./", reduce_dir="./", normalize=False, force=False) :
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
    if obstype == 'bias' :
        obstype_keyvalue = p['BIAS_OBSTYPE_KEYVALUE']
    elif obstype == 'flat' :
        obstype_keyvalue = p['FLAT_OBSTYPE_KEYVALUE']
    elif obstype == 'object' :
        obstype_keyvalue = p['OBJECT_OBSTYPE_KEYVALUE']
    else :
        print("obstype={} not recognized, setting to default = {}".format(obstype,obstype_keyvalue))
    
    if output == "" :
        output = os.path.join(reduce_dir, "master_{}.fits".format(obstype))
    
    # set master calib keyword in parameters
    p["master_{}".format(obstype)] = output
       
    # Skip if product already exists and reduction is not forced
    if os.path.exists(output) and not force :
        return p
    
    # set method to combine images
    method = p['CALIB_IMCOMBINE_METHOD']
    
    if inputlist == [] :
        # select FITS files in the minidata directory and build database
        main_fg = FitsFileGroup(location=data_dir, fits_ext=p['CALIB_WILD_CARDS'], ext=0)
        # print total number of files selected:
        print(f'Total files: {len(main_fg)}')
    
        # Filter files by header keywords
        filter_fg = main_fg.filtered({'obstype': obstype_keyvalue})
    else :
        filter_fg = FitsFileGroup(files=inputlist)
        
    # print total number of bias files selected
    print(f'{obstype} files: {len(filter_fg)}')

    # get frames
    frames = list(filter_fg.framedata(unit='adu', use_memmap_backend=p['USE_MEMMAP']))

    #extract gain from the first image
    if float(frames[0].header['GAIN']) != 0 :
        gain = float(frames[0].header['GAIN'])*u.electron/u.adu  # using quantities is better for safety
    else :
        gain = 3.3*u.electron/u.adu
    print('gain:', gain)
    
    # Perform gain calibration
    for i, frame in enumerate(frames):
        print(f'processing frame {i+1} of {len(frames)}')
        processing.cosmics_lacosmic(frame, inplace=True)
        processing.gain_correct(frame, gain, inplace=True)

    # combine
    master = imcombine(frames, method=method, use_memmap_backend=p['USE_MEMMAP'])

    # get statistics
    stats = master.statistics()
    
    norm_mean_value = master.mean()
    print('Normalization mean value:', norm_mean_value)

    data_units = 'electron'
    
    if normalize :
        master = imarith(master, norm_mean_value, '/')
        data_units = 'dimensionless'

    # write information into an info dict
    info = {'INCOMBME': ('{}'.format(method), 'imcombine method'),
        'INCOMBNI': (len(filter_fg), 'imcombine nimages'),
        'BUNIT': ('{}'.format(data_units), 'data units'),
        'DRSINFO': ('astropop', 'data reduction software'),
        'DRSROUT': ('master image', 'data reduction routine'),
        'NORMALIZ': (normalize, 'normalized master'),
        'NORMMEAN': (norm_mean_value.value,'normalization mean value in {}'.format(norm_mean_value.unit)),
        'MINVAL': (stats['min'].value,'minimum value in {}'.format(stats['min'].unit)),
        'MAXVAL': (stats['max'].value,'maximum value in {}'.format(stats['max'].unit)),
        'MEANVAL': (stats['mean'].value,'mean value in {}'.format(stats['mean'].unit)),
        'MEDIANVA': (stats['median'].value,'median value in {}'.format(stats['median'].unit)),
        'STDVALUE': (stats['std'].value,'standard deviation in {}'.format(stats['std'].unit))
        }
    
    # get data arrays
    img_data=np.array(master.data)
    err_data=np.array(master.get_uncertainty())
    mask_data=np.array(master.mask)
    
    # call function masteZero from sparc4_products to generate final product
    mastercal = s4p.masterCalibration(filter_fg.files, img_data=img_data, err_data=err_data, mask_data=mask_data, info=info, filename=output)
    
    return p
    

def old_reduce_science_images(p, inputlist, data_dir="./", reduce_dir="./", force=False, match_frames=False, stack_suffix="", output_stack="", polarimetry=False) :

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
    try :
        # load bias frame
        bias = s4p.getFrameFromMasterCalibration(p["master_bias"])
    except :
        print("WARNING: failed to read master bias, ignoring ...")
        
    try :
        # load flat frame
        flat = s4p.getFrameFromMasterCalibration(p["master_flat"])
    except :
        print("WARNING: failed to read master flat, ignoring ...")
        
    # select FITS files in the minidata directory and build database
    #main_fg = FitsFileGroup(location=data_dir, fits_ext=p['OBJECT_WILD_CARDS'], ext=p['SCI_EXT'])
    obj_fg = FitsFileGroup(files=inputlist)

    # print total number of object files selected
    print(f'OBJECT files: {len(obj_fg)}')

    # set base image as the reference image, which will be replaced later if run registering
    p['REF_IMAGE_INDEX'] = 0
    p['REFERENCE_IMAGE'] = obj_fg.files[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])
    
    # set output stack filename
    if output_stack == "" :
        output_stack = os.path.join(reduce_dir, '{}_stack.fits'.format(stack_suffix))
    p['OBJECT_STACK'] = output_stack
    
    print("Creating output list of processed science frames ... ")

    p['OBJECT_REDUCED_IMAGES'], obj_red_status = [], []
    
    for i in range(len(obj_fg.files)) :
        # get basename
        basename = os.path.basename(obj_fg.files[i])
        # create output name in the reduced dir
        output = os.path.join(reduce_dir, basename.replace(".fits","_proc.fits"))
        p['OBJECT_REDUCED_IMAGES'].append(output)
        
        red_status = False
        if not force :
            if os.path.exists(output) :
                red_status = True
        obj_red_status.append(red_status)
        
        print("{} of {} is reduced? {} -> {}".format(i+1, len(obj_fg.files), red_status, output))

    if not all(obj_red_status) or force:
        print("Loading science frames to memory ... ")
        # get frames
        frames = list(obj_fg.framedata(unit='adu', use_memmap_backend=p['USE_MEMMAP']))

        #extract gain from the first image
        if float(frames[0].header['GAIN']) != 0 :
            gain = float(frames[0].header['GAIN'])*u.electron/u.adu  # using quantities is better for safety
        else :
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
            print("Calibrating science frame {} of {} : {} ".format(i+1,len(frames),os.path.basename(obj_fg.files[i])))
            if not obj_red_status[i] or force:
                processing.cosmics_lacosmic(frame, inplace=True)
                processing.gain_correct(frame, gain, inplace=True)
                processing.subtract_bias(frame, bias, inplace=True)
                processing.flat_correct(frame, flat, inplace=True)
            else :
                pass

        if match_frames :
            print('Calculating offsets and selecting images for stack ... ')
            # run routine to select files that will be used for stack
            p = select_files_for_stack_and_get_shifts(p, frames, obj_fg.files, sort_method=p['METHOD_TO_SELECT_FILES_FOR_STACK'])
            
            info['REFIMG'] = (p['REFERENCE_IMAGE'], "reference image for stack")
            info['NIMGSTCK'] = (p['NFILES_FOR_STACK'], "number of images for stack")

            print('Registering science frames ... ')
            # Register images, generate global catalog and generate stack image
            p = run_register_frames(p, frames, obj_fg.files, info, output_stack=output_stack, force=force, polarimetry=polarimetry)
        
        # Perform aperture photometry and store reduced data into products
        for i, frame in enumerate(frames):
        
            info['XSHIFT'] = (0.,"register x shift (pixel)")
            info['XSHIFTST'] = ("OK","x shift status")
            info['YSHIFT'] = (0.,"register y shift (pixel)")
            info['YSHIFTST'] = ("OK","y shift status")
            if match_frames :
                if np.isfinite(p["XSHIFTS"][i]) :
                    info['XSHIFT'] = (p["XSHIFTS"][i],"register x shift (pixel)")
                else :
                    info['XSHIFTST'] = ("UNDEFINED","x shift status")

                if np.isfinite(p["YSHIFTS"][i]) :
                    info['YSHIFT'] = (p["YSHIFTS"][i],"register y shift (pixel)")
                else :
                    info['YSHIFTST'] = ("UNDEFINED","y shift status")

            if not obj_red_status[i] or force:
                # get data arrays
                img_data=np.array(frame.data)
                err_data=np.array(frame.get_uncertainty())
                mask_data=np.array(frame.mask)

                try :
                    # make catalog
                    if match_frames :
                        p, frame_catalogs = build_catalogs(p, img_data, deepcopy(p["CATALOGS"]), xshift=p["XSHIFTS"][i], yshift=p["YSHIFTS"][i], polarimetry=polarimetry)
                    else :
                        p, frame_catalogs = build_catalogs(p, img_data, polarimetry=polarimetry)
                except :
                    print("WARNING: could not build frame catalog")
                    # set local
                    frame_catalogs = []

                print("Saving frame {} of {}:".format(i+1,len(frames)),obj_fg.files[i],'->',p['OBJECT_REDUCED_IMAGES'][i])

                frame_wcs_header = deepcopy(p['WCS_HEADER'])

                if np.isfinite(p["XSHIFTS"][i]) :
                    frame_wcs_header['CRPIX1'] = frame_wcs_header['CRPIX1'] + p["XSHIFTS"][i]
                if np.isfinite(p["YSHIFTS"][i]) :
                    frame_wcs_header['CRPIX2'] = frame_wcs_header['CRPIX2'] + p["YSHIFTS"][i]

                # call function to generate final product
                # for light products
                s4p.scienceImageLightProduct(obj_fg.files[i], img_data=img_data, info=info, catalogs=frame_catalogs, polarimetry=polarimetry, filename=p['OBJECT_REDUCED_IMAGES'][i], catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=frame_wcs_header, time_key=p["TIME_KEY"])
                # for more complete products with an error and mask extensions
                #s4p.scienceImageProduct(obj_fg.files[i], img_data=img_data, err_data=err_data, mask_data=mask_data, info=info, catalogs=frame_catalogs, polarimetry=polarimetry, filename=p['OBJECT_REDUCED_IMAGES'][i], catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=frame_wcs_header, time_key=p["TIME_KEY"])

    return p


def reduce_science_images(p, inputlist, data_dir="./", reduce_dir="./", match_frames=True, ref_img="", force=False, polarimetry=False, ra="", dec="") :

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
    try :
        # load bias frame
        bias = s4p.getFrameFromMasterCalibration(p["master_bias"])
    except :
        print("WARNING: failed to read master bias, ignoring ...")
        
    try :
        # load flat frame
        flat = s4p.getFrameFromMasterCalibration(p["master_flat"])
    except :
        print("WARNING: failed to read master flat, ignoring ...")
    
    # save original input list of files
    p['INPUT_LIST_OF_FILES'] = deepcopy(inputlist)
    # check whether the input reference image is in the input list
    if ref_img not in inputlist :
        # add ref image to the list
        inputlist = [ref_img] + inputlist
        
    # make sure to get the correct index for the reference image in the new list
    p['REF_IMAGE_INDEX'] = inputlist.index(p['REFERENCE_IMAGE'])

    # select FITS files in the minidata directory and build database
    #main_fg = FitsFileGroup(location=data_dir, fits_ext=p['OBJECT_WILD_CARDS'], ext=p['SCI_EXT'])
    obj_fg = FitsFileGroup(files=inputlist)

    # print total number of object files selected
    print(f'OBJECT files: {len(obj_fg)}')
    
    print("Creating output list of processed science frames ... ")
    
    obj_red_images, obj_red_status = [], []

    for i in range(len(obj_fg.files)) :
        # get basename
        basename = os.path.basename(obj_fg.files[i])
        # create output name in the reduced dir
        output = os.path.join(reduce_dir, basename.replace(".fits","_proc.fits"))
        obj_red_images.append(output)
        
        red_status = False
        if not force :
            if os.path.exists(output) :
                red_status = True
        obj_red_status.append(red_status)
        
        print("{} of {} is reduced? {} -> {}".format(i+1, len(obj_fg.files), red_status, output))

    if not all(obj_red_status) or force:
        print("Loading science frames to memory ... ")
        # get frames
        frames = list(obj_fg.framedata(unit='adu', use_memmap_backend=p['USE_MEMMAP']))

        #extract gain from the first image
        if float(frames[0].header['GAIN']) != 0 :
            gain = float(frames[0].header['GAIN'])*u.electron/u.adu  # using quantities is better for safety
        else :
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
            print("Calibrating science frame {} of {} : {} ".format(i+1,len(frames),os.path.basename(obj_fg.files[i])))
            if not obj_red_status[i] or force:
                processing.cosmics_lacosmic(frame, inplace=True)
                processing.gain_correct(frame, gain, inplace=True)
                processing.subtract_bias(frame, bias, inplace=True)
                processing.flat_correct(frame, flat, inplace=True)
            else :
                pass

        print('Calculating offsets ... ')
        p = compute_offsets(p, frames, obj_fg.files, auto_ref_selection=False)
        
        # write ref image to header
        info['REFIMG'] = (p['REFERENCE_IMAGE'], "reference image")

        # Perform aperture photometry and store reduced data into products
        for i, frame in enumerate(frames):

            info['XSHIFT'] = (0.,"register x shift (pixel)")
            info['XSHIFTST'] = ("OK","x shift status")
            info['YSHIFT'] = (0.,"register y shift (pixel)")
            info['YSHIFTST'] = ("OK","y shift status")
            if match_frames :
                if np.isfinite(p["XSHIFTS"][i]) :
                    info['XSHIFT'] = (p["XSHIFTS"][i],"register x shift (pixel)")
                else :
                    info['XSHIFTST'] = ("UNDEFINED","x shift status")

                if np.isfinite(p["YSHIFTS"][i]) :
                    info['YSHIFT'] = (p["YSHIFTS"][i],"register y shift (pixel)")
                else :
                    info['YSHIFTST'] = ("UNDEFINED","y shift status")

            if not obj_red_status[i] or force:
                # get data arrays
                img_data=np.array(frame.data)
                try :
                    # make catalog
                    if match_frames and "CATALOGS" in p.keys() :
                        p, frame_catalogs = build_catalogs(p, img_data, deepcopy(p["CATALOGS"]), xshift=p["XSHIFTS"][i], yshift=p["YSHIFTS"][i], polarimetry=polarimetry)
                    else :
                        p, frame_catalogs = build_catalogs(p, img_data, polarimetry=polarimetry)
                except :
                    print("WARNING: could not build frame catalog.")
                    # set local
                    frame_catalogs = []

                print("Saving frame {} of {}:".format(i+1,len(frames)),obj_fg.files[i],'->',obj_red_images[i])

                frame_wcs_header = deepcopy(p['WCS_HEADER'])

                if np.isfinite(p["XSHIFTS"][i]) :
                    frame_wcs_header['CRPIX1'] = frame_wcs_header['CRPIX1'] + p["XSHIFTS"][i]
                if np.isfinite(p["YSHIFTS"][i]) :
                    frame_wcs_header['CRPIX2'] = frame_wcs_header['CRPIX2'] + p["YSHIFTS"][i]

                # call function to generate final product
                # for light products
                s4p.scienceImageLightProduct(obj_fg.files[i], img_data=img_data, info=info, catalogs=frame_catalogs, polarimetry=polarimetry, filename=obj_red_images[i], catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=frame_wcs_header, time_key=p["TIME_KEY"], ra=ra, dec=dec)
                
    if 'OBJECT_REDUCED_IMAGES' not in p.keys() :
        p['OBJECT_REDUCED_IMAGES'] = obj_red_images
    else :
        for i in range(len(obj_red_images)) :
            if obj_red_images[i] not in p['OBJECT_REDUCED_IMAGES'] :
                p['OBJECT_REDUCED_IMAGES'].append(obj_red_images[i])
    return p

    
def stack_science_images(p, inputlist, data_dir="./", reduce_dir="./", force=False, stack_suffix="", output_stack="", polarimetry=False) :

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
            
    data_dir : str, optional
        String to define the directory path to the raw data
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
    if output_stack == "" :
        output_stack = os.path.join(reduce_dir, '{}_stack.fits'.format(stack_suffix))
    p['OBJECT_STACK'] = output_stack
    
    if os.path.exists(p['OBJECT_STACK']) and not force :
        print("There is already a stack image :", p['OBJECT_STACK'])
        
        stack_hdulist = fits.open(p['OBJECT_STACK'])
        hdr = stack_hdulist[0].header
        p['REFERENCE_IMAGE'] = hdr['REFIMG']
        p['REF_IMAGE_INDEX'] = inputlist.index(p['REFERENCE_IMAGE'])
        p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])
        
        p["CATALOGS"] = s4p.readScienceImagCatalogs(p['OBJECT_STACK'])

        return p
    
    # read master calibration files
    try :
        # load bias frame
        bias = s4p.getFrameFromMasterCalibration(p["master_bias"])
    except :
        print("WARNING: failed to read master bias, ignoring ...")
        
    try :
        # load flat frame
        flat = s4p.getFrameFromMasterCalibration(p["master_flat"])
    except :
        print("WARNING: failed to read master flat, ignoring ...")
    
    # set base image as the reference image, which will be replaced
    # later if run "select_files_for_stack"
    p['REF_IMAGE_INDEX'] = 0
    p['REFERENCE_IMAGE'] = inputlist[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])
        
    # first select best files for stack
    p = select_files_for_stack(p, inputlist, saturation_limit=p['SATURATION_LIMIT'], imagehdu=0)

    # select FITS files in the minidata directory and build database
    obj_fg = FitsFileGroup(files=p['SELECTED_FILES_FOR_STACK'])

    # print total number of object files selected
    print(f'OBJECT files: {len(obj_fg)}')

    print("Loading science frames to memory ... ")
    # get frames
    frames = list(obj_fg.framedata(unit='adu', use_memmap_backend=p['USE_MEMMAP']))

    #extract gain from the first image
    if float(frames[0].header['GAIN']) != 0 :
        gain = float(frames[0].header['GAIN'])*u.electron/u.adu  # using quantities is better for safety
    else :
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
            'NIMGSTCK': (p['NFILES_FOR_STACK'], 'number of images for stack')
            }

    print('Calibrating science frames (CR, gain, bias, flat) ... ')

    # Perform calibration
    for i, frame in enumerate(frames):
        print("Calibrating science frame {} of {} : {} ".format(i+1,len(frames),os.path.basename(obj_fg.files[i])))
        processing.cosmics_lacosmic(frame, inplace=True)
        processing.gain_correct(frame, gain, inplace=True)
        processing.subtract_bias(frame, bias, inplace=True)
        processing.flat_correct(frame, flat, inplace=True)

    print('Registering science frames and stacking them ... ')
    
    p['SELECTED_FILE_INDICES_FOR_STACK'] = np.arange(p['NFILES_FOR_STACK'])

    # Register images, generate global catalog and generate stack image
    p = run_register_frames(p, frames, obj_fg.files, info, output_stack=output_stack, force=force, polarimetry=polarimetry)

    return p


def select_files_for_stack(p, inputlist, saturation_limit=32768, imagehdu=0) :
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
    for i in range(len(inputlist)) :
        img = fits.getdata(inputlist[i],imagehdu)
        keep = img < saturation_limit
        bkg = np.nanmedian(img[keep])
        maximgflux = np.nanmax(img[keep]) - bkg
        if maximgflux > maxflux :
            maxflux = maximgflux
            ref_img_idx = i
        peaks.append(maximgflux)
        #print(i, inputlist[i], '-> ', bkg, (maximgflux - bkg))
    peaks = np.array(peaks)
    
    p['REF_IMAGE_INDEX'] = ref_img_idx
    p['REFERENCE_IMAGE'] = inputlist[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])
    
    #plt.plot(peaks)
    #plt.show()
    print(p['REF_IMAGE_INDEX'], "Reference image: {}".format(p['REFERENCE_IMAGE']))
    
    sort = np.flip(np.argsort(peaks))
    
    sorted_files = []
    newsort = []
    # first add reference image
    sorted_files.append(p['REFERENCE_IMAGE'])
    newsort.append(p['REF_IMAGE_INDEX'])
    # then add all valid images to the list of images for stack
    for i in sort :
        if inputlist[i] != p['REFERENCE_IMAGE'] :
            sorted_files.append(inputlist[i])
            newsort.append(i)
    newsort = np.array(newsort)
    
    # Now select up to <N files for stack as defined in the parameters file and save list to the param dict
    if len(sorted_files) > p['NFILES_FOR_STACK'] :
        p['SELECTED_FILES_FOR_STACK'] = sorted_files[:p['NFILES_FOR_STACK']]
        #p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort[:p['NFILES_FOR_STACK']]
    else :
        p['NFILES_FOR_STACK'] = len(sorted_files)
        p['SELECTED_FILES_FOR_STACK'] = sorted_files
        #p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort

    return p


def compute_offsets(p, frames, obj_files, auto_ref_selection=False) :
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

    if auto_ref_selection :
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

    print("Computing offsets with respect to the reference image: index={} -> {}".format(p['REF_IMAGE_INDEX'], obj_files[p['REF_IMAGE_INDEX']]))
    
    # get x and y shifts of all images with respect to the first image
    shift_list = compute_shift_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=p['REF_IMAGE_INDEX'], skip_failure=True)

    # store shifts in x and y np arrays
    x, y = np.array([]), np.array([])
    for i in range(len(shift_list)) :
        x = np.append(x, shift_list[i][0])
        y = np.append(y, shift_list[i][1])
        #print(i,shift_list[i][0],shift_list[i][1])

    # store shifts
    p["XSHIFTS"] = x
    p["YSHIFTS"] = y

    return p


def select_files_for_stack_and_get_shifts(p, frames, obj_files, sort_method='MAX_FLUXES', correct_shifts=False, plot=False) :
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
    shift_list = compute_shift_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=p['REF_IMAGE_INDEX'], skip_failure=True)

    # store shifts in x and y np arrays
    x, y = np.array([]), np.array([])
    for i in range(len(shift_list)) :
        x = np.append(x, shift_list[i][0])
        y = np.append(y, shift_list[i][1])
        #print(i,shift_list[i][0],shift_list[i][1])

    # store shifts
    p["XSHIFTS"] = x
    p["YSHIFTS"] = y

    if sort_method == 'MAX_FLUXES' :
        sort = np.flip(np.argsort(peaksnr))

    elif sort_method == 'MIN_SHIFTS' :
        # calculate median shifts
        median_x = np.nanmedian(x)
        median_y = np.nanmedian(y)
        # calculate relative distances to the median values
        dist = np.sqrt((x - median_x)**2 + (y - median_y)**2)
        # sort in crescent order
        sort = np.argsort(dist)
        
        if plot :
            indices = np.arange(len(sort))
            plt.plot(indices,dist,'ko')
            if len(sort) > p['NFILES_FOR_STACK'] :
                plt.plot(indices[sort][:p['NFILES_FOR_STACK']], dist[sort][:p['NFILES_FOR_STACK']],'rx')
            else :
                plt.plot(indices,dist,'rx')
            plt.xlabel("Image index")
            plt.ylabel("Distance to median position (pixel)")
            plt.show()
    else :
        print("ERROR: sort_method = {} not recognized, select a valid method.".format(sort_method))
        exit()    # select files in crescent order based on the distance to the median position
        
    sorted_files = []
    newsort = []
    
    # first add reference image
    sorted_files.append(p['REFERENCE_IMAGE'])
    newsort.append(p['REF_IMAGE_INDEX'])

    # then add all valid images to the list of images for stack
    for i in sort :
        #print(obj_files[i],p["XSHIFTS"][i],p["YSHIFTS"][i],p["PEAKS"][i],p["PEAK_SNR"][i],p["MEAN_BKG"][i])
        if np.isfinite(p["XSHIFTS"][i]) and np.isfinite(p["YSHIFTS"][i]) and p["PEAK_SNR"][i] > 1. and i != p['REF_IMAGE_INDEX'] :
            sorted_files.append(obj_files[i])
            newsort.append(i)
    newsort = np.array(newsort)
    
    # Now select up to <N files for stack as defined in the parameters file and save list to the param dict
    if len(sorted_files) > p['NFILES_FOR_STACK'] :
        p['SELECTED_FILES_FOR_STACK'] = sorted_files[:p['NFILES_FOR_STACK']]
        p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort[:p['NFILES_FOR_STACK']]
        """
        if correct_shifts :
            # Correct shifts to match the new reference image for the catalog
            p["XSHIFTS"] -= p["XSHIFTS"][newsort[0]]
            p["YSHIFTS"] -= p["YSHIFTS"][newsort[0]]
            """
    else :
        p['NFILES_FOR_STACK'] = len(sorted_files)
        p['SELECTED_FILES_FOR_STACK'] = sorted_files
        p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort

    return p
    
    
def run_register_frames(p, inframes, inobj_files, info, output_stack="", force=False, maxnsources=0, polarimetry=False) :
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
    for i in p['SELECTED_FILE_INDICES_FOR_STACK'] :
        print(i,inobj_files[i])
        frames.append(deepcopy(inframes[i]))
        obj_files.append(inobj_files[i])
    
    stack_method = p['SCI_STACK_METHOD']
    
    #shift_list = compute_shift_list(frames, algorithm='asterism-matching', ref_image=0)
    #print(shift_list)

    # register frames
    registered_frames = register_framedata_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=0, inplace=False)
        
    # stack all object files
    combined = imcombine(registered_frames, method=stack_method)
    
    # get stack data
    img_data = np.array(combined.data, dtype=float)
    err_data = np.array(combined.get_uncertainty())
    mask_data = np.array(combined.mask)
        
    # get an aperture that's 2 x fwhm measure on the stacked image
    p = calculate_aperture_radius(p, img_data)
       
    # generate catalog
    p, stack_catalogs = build_catalogs(p, img_data, maxnsources=maxnsources, polarimetry=polarimetry, stackmode=True)

    # set master catalogs
    p["CATALOGS"] = stack_catalogs

    # save stack product
    if output_stack != "" :
        if not os.path.exists(output_stack) or force:
            # for light products
            s4p.scienceImageLightProduct(obj_files[0], img_data=img_data, info=info, catalogs=p["CATALOGS"], polarimetry=polarimetry, filename=output_stack, catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=p['WCS_HEADER'], time_key=p["TIME_KEY"])
            # for more complete products with an error and mask extensions
            #s4p.scienceImageProduct(obj_files[0], img_data=img_data, err_data=err_data, mask_data=mask_data, info=info, catalogs=p["CATALOGS"], polarimetry=polarimetry, filename=output_stack, catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=p['WCS_HEADER'], time_key=p["TIME_KEY"])
    return p
    
   
def uflux_to_magnitude(uflux) :
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

    try :
        return -2.5 * umath.log10(uflux)
    except :
        return ufloat(np.nan, np.nan)
    
def flux_to_magnitude(flux) :
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
    try :
        return -2.5 * np.log10(flux)
    except :
        return np.nan


def get_peaks_and_fwhms(data, x, y, box_size=10, model='gaussian') :
    """ Pipeline tool to calculate peak fluxes and fwhms within boxes
         around each point in a list of coordinates
    
    Parameters
    ----------
    data : numpy.ndarray (n x m)
        float array containing the image data
    x : numpy.ndarray (N sources)
        float array containing the list of x-coord data
    y : numpy.ndarray (N sources)
        float array containing the list of y-coord data
    box_size : int (default: 10)
        aresta of squared box around each source
    model : str (default: 'gaussian')
        model type for the calculation of FWHM
    Returns
    -------
    peaks : numpy.ndarray (N sources)
        peak fluxes for each source
    fwhms : numpy.ndarray (N sources)
        full width at half maximum (fwhm) for each source
    """

    indices = np.indices(data.shape)
    
    rects = [trim_array(data, box_size, (xi, yi), indices=indices)
             for xi, yi in zip(x, y)]
             
    fwhms = [_fwhm_loop(model, d[0], d[1], d[2], xi, yi)
            for d, xi, yi in zip(rects, x, y)]
    
    peaks = np.array([])
    for rec in rects :
        peaks = np.append(peaks, np.nanmax(rec))

    return peaks, fwhms


def calculate_aperture_radius(p, data) :
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
    sources = starfind(data, threshold=p['PHOT_THRESHOLD'], background=bkg, noise=rms)

    # get fwhm
    fwhm = sources.meta['astropop fwhm']
    
    p["PHOT_APERTURE_RADIUS"] = p["PHOT_APERTURE_N_X_FWHM"] * fwhm
    p["PHOT_SKYINNER_RADIUS"] = p["PHOT_SKYINNER_N_X_FWHM"] * fwhm
    p["PHOT_SKYOUTER_RADIUS"] = p["PHOT_SKYOUTER_N_X_FWHM"] * fwhm

    return p


def run_aperture_photometry(img_data, x, y, aperture_radius, r_ann, output_mag=True, sortbyflux=True) :
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
    Returns
        x, y, mag, mag_error, smag, smag_error, flags
    -------
     :
    """
        
    # perform aperture photometry
    ap_phot = aperture_photometry(img_data, x, y, r=aperture_radius, r_ann=r_ann)
    
    # I cannot get round and sharp because they may not match master catalog data
    #ap_phot['round'], ap_phot['sharp'] = sources['round'], sources['sharp']
    
    # sort table in flux
    if sortbyflux :
        ap_phot.sort('flux')
        # reverse, to start with the highest flux
        ap_phot.reverse()

    x,y = np.array(ap_phot['x']), np.array(ap_phot['y'])
    flux, flux_error = np.array(ap_phot['flux']), np.array(ap_phot['flux_error'])
    sky = np.array(ap_phot['sky'])
    #round = np.array(ap_phot['round'])
    #sharp = np.array(ap_phot['sharp'])
    flags = np.array(ap_phot['flags'])
    
    if output_mag :
        mag, mag_error = np.full_like(flux,np.nan),np.full_like(flux,np.nan)
        smag, smag_error = np.full_like(flux,np.nan),np.full_like(flux,np.nan)
        for i in range(len(flux)) :
            umag = uflux_to_magnitude(ufloat(flux[i],flux_error[i]))
            mag[i], mag_error[i] = umag.nominal_value, umag.std_dev
            #detnoise = np.sqrt(flux_error[i] * flux_error[i] - flux[i])
            #uskymag = uflux_to_magnitude(ufloat(sky[i],np.sqrt(sky[i]+detnoise*detnoise)))
            uskymag = uflux_to_magnitude(ufloat(sky[i],np.sqrt(sky[i])))
            smag[i], smag_error[i] = uskymag.nominal_value, uskymag.std_dev
        
        return x, y, mag, mag_error, smag, smag_error, flags

    else :

        return x, y, flux, flux_error, sky, np.sqrt(sky), flags


def read_catalog_coords(catalog) :
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
    
    for star in catalog.keys() :
        ra = np.append(ra,catalog[star][1])
        dec = np.append(dec,catalog[star][2])
        x = np.append(x,catalog[star][3])
        y = np.append(y,catalog[star][4])

    return ra, dec, x, y

def set_wcs(p) :
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
    ra, dec = p['REF_OBJECT_HEADER']['RA'].split(":"), p['REF_OBJECT_HEADER']['DEC'].split(":")
            
    ra_str = '{:02d}h{:02d}m{:.2f}s'.format(int(ra[0]),int(ra[1]),float(ra[2]))
    dec_str = '{:02d}d{:02d}m{:.2f}s'.format(int(dec[0]),int(dec[1]),float(dec[2]))
    #print(ra_str, dec_str)
            
    coord = SkyCoord(ra_str, dec_str, frame='icrs')
    ra_deg, dec_deg = coord.ra.degree, coord.dec.degree
    p['RA_DEG'], p['DEC_DEG'] = ra_deg, dec_deg
    p['WCS'] = WCS(fits.getheader(p["ASTROM_REF_IMG"],0),naxis=2)
    p['WCS_HEADER'] = p['WCS'].to_header(relax=True)
    p['WCS_HEADER']['CRVAL1'] = ra_deg
    p['WCS_HEADER']['CRVAL2'] = dec_deg
    #p['WCS_HEADER']['LATPOLE'] = 0
    #p['WCS_HEADER']['LONPOLE'] = 180
    del p['WCS_HEADER']['DATE-OBS']
    del p['WCS_HEADER']['MJD-OBS']

    return p

def generate_catalogs(p, data, sources, fwhm, catalogs=[], catalogs_label='', aperture_radius=10, r_ann=(25,50), sortbyflux=True, maxnsources=0, polarimetry=False, use_e_beam_for_astrometry=False, solve_astrometry=False) :
    
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

    if polarimetry :
        catalogs.append({})
        catalogs.append({})

        # no input catalogs, then create new ones
        dx, dy = estimate_dxdy(sources['x'], sources['y'])
        pairs = match_pairs(sources['x'], sources['y'], dx, dy, tolerance=p["MATCH_PAIRS_TOLERANCE"])
    
        sources_table = Table()

        sources_table['star_index'] = np.arange(len(pairs))+1
        sources_table['x_o'] = sources['x'][pairs['o']]
        sources_table['y_o'] = sources['y'][pairs['o']]
        sources_table['x_e'] = sources['x'][pairs['e']]
        sources_table['y_e'] = sources['y'][pairs['e']]

        #s4plt.plot_sci_polar_frame(data, bkg, sources_table)
        #print("sources:\n",sources)
        #print("\n\nsources_table:\n",sources_table)

        xo, yo, mago, mago_error, smago, smago_error, flagso = run_aperture_photometry(data, sources_table['x_o'], sources_table['y_o'], aperture_radius, r_ann, output_mag=True, sortbyflux=False)

        sorted = np.full_like(xo, True, dtype=bool)
        if sortbyflux :
            sorted = np.argsort(mago)

        xe, ye, mage, mage_error, smage, smage_error, flagse = run_aperture_photometry(data, sources_table['x_e'], sources_table['y_e'], aperture_radius, r_ann, output_mag=True, sortbyflux=False)

        xo, yo = xo[sorted], yo[sorted]
        mago, mago_error = mago[sorted], mago_error[sorted]
        smago, smago_error = smago[sorted], smago_error[sorted]
        flagso = flagso[sorted]
 
        xe, ye = xe[sorted], ye[sorted]
        mage, mage_error = mage[sorted], mage_error[sorted]
        smage, smage_error = smage[sorted], smage_error[sorted]
        flagse = flagse[sorted]

        fwhmso, fwhmse = np.full_like(mago,fwhm), np.full_like(mage,fwhm)

        if use_e_beam_for_astrometry :
            xs_for_astrometry = xo
            ys_for_astrometry = yo
        else :
            xs_for_astrometry = xe
            ys_for_astrometry = ye
            
        if solve_astrometry :
            if use_e_beam_for_astrometry :
                fluxes_for_astrometry = 10**(-0.4*mage)
            else :
                fluxes_for_astrometry = 10**(-0.4*mago)
                
            h, w = np.shape(data)
            
            #print ("I'm trying now to solve astrometry ...")
            #print("INPUT parameters: ",xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, h, w, p['REF_OBJECT_HEADER'], {'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE']-0.015, 'scale-units': 'arcsecperpix', 'scale-high':p['PLATE_SCALE']+0.015, 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER']})
            try :
                # Solve astrometry
                #p['WCS'] = solve_astrometry_xy(xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, image_height=h, image_width=w, image_header=p['REF_OBJECT_HEADER'], image_params={'ra': p['RA_DEG'],'dec': p['DEC_DEG'],'pltscl': p['PLATE_SCALE']}, return_wcs=True)
                #solution = solve_astrometry_xy(xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, height=h, width=w, image_header=p['REF_OBJECT_HEADER'], options={'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE']-0.02, 'scale-units': 'arcsecperpix', 'scale-high':p['PLATE_SCALE']+0.02, 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER']})
                solution = solve_astrometry_xy(xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, w, h, options={'ra': p['RA_DEG'],'dec':p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE']-0.02,'scale-high': p['PLATE_SCALE']+0.02,'scale-units': 'arcsecperpix','crpix-center': 1, 'tweak-order': p['TWEAK_ORDER']})
                
                p['WCS'] = solution.wcs
                
                p['WCS_HEADER'] = p['WCS'].to_header(relax=True)

            except :
                print("WARNING: could not solve astrometry, using WCS from database")

        ras, decs = p['WCS'].all_pix2world(xs_for_astrometry, ys_for_astrometry, 0)
        
        nsources = len(mago)
        if maxnsources :
            nsources = maxnsources

        # save photometry data into the catalogs
        for i in range(nsources) :
            catalogs[current_catalogs_len]["{}".format(i)] = (i, ras[i], decs[i], xo[i], yo[i], fwhmso[i], fwhmso[i], mago[i], mago_error[i], smago[i], smago_error[i], aperture_radius, flagso[i])
            catalogs[current_catalogs_len+1]["{}".format(i)] = (i, ras[i], decs[i], xe[i], ye[i], fwhmse[i], fwhmse[i], mage[i], mage_error[i], smage[i], smage_error[i], aperture_radius, flagse[i])
    else :
        catalogs.append({})

        #x, y = np.array(sources['x']), np.array(sources['y'])
        x, y, mag, mag_error, smag, smag_error, flags = run_aperture_photometry(data, sources['x'], sources['y'], aperture_radius, r_ann, output_mag=True, sortbyflux=sortbyflux)

        fwhms = np.full_like(mag,fwhm)

        if solve_astrometry :
            fluxes_for_astrometry = 10**(-0.4*mag)
            h, w = np.shape(data)
            
            try :
                #image_header=p['REF_OBJECT_HEADER']
                solution = solve_astrometry_xy(x, y, fluxes_for_astrometry, w, h, options={'ra': p['RA_DEG'],'dec':p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE']-0.02,'scale-high': p['PLATE_SCALE']+0.02,'scale-units': 'arcsecperpix','crpix-center': 1, 'tweak-order': p['TWEAK_ORDER']})
                p['WCS'] = solution.wcs
                p['WCS_HEADER'] = p['WCS'].to_header(relax=True)
            except :
                print("WARNING: could not solve astrometry, using WCS from database")

        ras, decs = p['WCS'].all_pix2world(x, y, 0)

        nsources = len(mag)
        if maxnsources :
            nsources = maxnsources

        # save photometry data into the catalog
        for i in range(nsources) :
            catalogs[current_catalogs_len]["{}".format(i)] = (i, ras[i], decs[i], x[i], y[i], fwhms[i], fwhms[i], mag[i], mag_error[i], smag[i], smag_error[i], aperture_radius, flags[i])

    return catalogs


def set_sky_aperture(p, aperture_radius) :
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
    if r_ann[0] < aperture_radius + p['PHOT_MIN_OFFSET_FOR_SKYINNERRADIUS'] :
        r_in_ann = aperture_radius + p['PHOT_MIN_OFFSET_FOR_SKYINNERRADIUS']
        
    r_out_ann = r_ann[1]
    if r_out_ann < r_in_ann + p['PHOT_MIN_OFFSET_FOR_SKYOUTERRADIUS']:
        r_out_ann = r_in_ann + p['PHOT_MIN_OFFSET_FOR_SKYOUTERRADIUS']
    
    r_ann = (r_in_ann, r_out_ann)

    return r_ann


def build_catalogs(p, data, catalogs=[], xshift=0., yshift=0., solve_astrometry=True, maxnsources=0, polarimetry=False, stackmode=False) :
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
    #hdul = fits.open(image_name, mode = "readonly")
    #data = np.array(hdul[0].data, dtype=float)
    
    # calculate background
    bkg, rms = background(data, global_bkg=False)
        
    # detect sources
    sources = starfind(data, threshold=p["PHOT_THRESHOLD"], background=bkg, noise=rms)
    
    # get fwhm
    fwhm = sources.meta['astropop fwhm']
    
    # set aperture radius from
    aperture_radius = p['PHOT_FIXED_APERTURE']
    r_ann = set_sky_aperture(p, aperture_radius)

    p = set_wcs(p)
    
    #print("******* DEBUG ASTROMETRY **********")
    #print("RA_DEG={} DEC_DEG={}".format(p['RA_DEG'], p['DEC_DEG']))
    #print("WCS:\n{}".format(p['WCS']))
    
    if stackmode :
        catalogs = []

    if catalogs == [] :
        print("Creating new catalog of detected sources:".format(xshift,yshift))
        
        # print sources table
        #print(sources)

        #print("Running aperture photometry with aperture_radius={} r_ann={}".format(aperture_radius,r_ann))
        catalogs = generate_catalogs(p, data, sources, fwhm, catalogs, aperture_radius=aperture_radius, r_ann=r_ann, polarimetry=polarimetry, solve_astrometry=True)

        if p['MULTI_APERTURES'] :
            for i in range(len(p['PHOT_APERTURES'])) :
                aperture_radius = p['PHOT_APERTURES'][i]
                r_ann = set_sky_aperture(p, aperture_radius)
                #print("Running aperture photometry with aperture_radius={} r_ann={}".format(aperture_radius,r_ann))
                catalogs = generate_catalogs(p, data, sources, fwhm, catalogs, aperture_radius=aperture_radius, r_ann=r_ann, polarimetry=polarimetry, solve_astrometry=False)
    else :
    
        print("Running aperture photometry for exiting catalogs offset by dx={} dy={}".format(xshift,yshift))

        for j in range(len(catalogs)) :
            # load coordinates from an input catalog
            ras, decs, x, y = read_catalog_coords(catalogs[j])
            
            # apply shifts
            if np.isfinite(xshift) :
                x += xshift
            if np.isfinite(yshift) :
                y += yshift

            aperture_radius = catalogs[j]['0'][11]
            r_ann = set_sky_aperture(p, aperture_radius)
            
            #print("Running aperture photometry for catalog={} xshift={} yshift={} with aperture_radius={} r_ann={}".format(j,xshift,yshift,aperture_radius,r_ann))

            # run aperture photometry
            x, y, mag, mag_error, smag, smag_error, flags = run_aperture_photometry(data, x, y, aperture_radius, r_ann, output_mag=True, sortbyflux=False)
            fwhms = np.full_like(mag,fwhm)

            # save data back into the catalog
            for i in range(len(mag)) :
                catalogs[j]["{}".format(i)] = (i, ras[i], decs[i], x[i], y[i], fwhms[i], fwhms[i], mag[i], mag_error[i], smag[i], smag_error[i], aperture_radius, flags[i])

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
                    ref_catalog_name="",
                    catalog_names=[],
                    polarimetry=False,
                    force=True) :
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
    ref_catalog_name : str (optional)
        reference catalog name. Time and coordinates data will be taken from this catalog only
    catalog_names : list (optional)
        list of catalog names to be included in the final data product. The less the faster. Default is [], which means all catalogs.
    polarimetry : bool
        whether or not polarimetry mode

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

    if catalog_names == [] :
        for hdu in hdul :
            if hdu.name != "PRIMARY" :
                catalog_names.append(hdu.name)
    
    # print list of catalogs
    #print(catalog_names)
    
    # if reference catalog name is not given then assign first catalog as the reference
    if ref_catalog_name == "" :
        ref_catalog_name = catalog_names[0]

    # read catalog of base image to figure out number of sources
    catdata = s4p.readPhotTimeSeriesData(sci_list,
                                    catalog_key=ref_catalog_name,
                                    longitude=longitude,
                                    latitude=latitude,
                                    altitude=altitude,
                                    time_keyword=time_key,
                                    time_format=time_format,
                                    time_scale=time_scale,
                                    gettimedata=True,
                                    getcoordsdata=True,
                                    getphotdata=False)
    # get time data
    tsdata["TIME"] = deepcopy(catdata["TIME"])
    
    # get coords data
    tsdata["X"] = deepcopy(catdata["X"])
    tsdata["Y"] = deepcopy(catdata["Y"])
    tsdata["RA"] = deepcopy(catdata["RA"])
    tsdata["DEC"] = deepcopy(catdata["DEC"])
    tsdata["FWHM"] = deepcopy(catdata["FWHM"])

    del catdata
    
    # loop below to get photometry data for all catalogs
    for key in catalog_names :

        print("Packing time series data for catalog: {}".format(key))

        catdata = s4p.readPhotTimeSeriesData(sci_list,
                                catalog_key=key,
                                longitude=longitude,
                                latitude=latitude,
                                altitude=altitude,
                                time_keyword=time_key,
                                time_format=time_format,
                                time_scale=time_scale,
                                gettimedata=False,
                                getcoordsdata=False,
                                getphotdata=True)
        tsdata[key] = {}
        
        tsdata[key]["MAG"] = deepcopy(catdata["MAG"])
        tsdata[key]["EMAG"] = deepcopy(catdata["EMAG"])
        tsdata[key]["SKYMAG"] = deepcopy(catdata["SKYMAG"])
        tsdata[key]["ESKYMAG"] = deepcopy(catdata["ESKYMAG"])
        tsdata[key]["FLAG"] = deepcopy(catdata["FLAG"])

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
    source = SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg), frame='icrs', equinox=equinox)
    info['RA'] = (source.ra.value, '[DEG] RA of object of interest')
    info['DEC'] = (source.dec.value, '[DEG] DEC of object of interest')
    info['RADESYS'] = ('ICRS    ', 'reference frame of celestial coordinates')
    info['EQUINOX'] = (2000.0, 'equinox of celestial coordinate system')
    info['PHZEROP'] = (0., '[mag] photometric zero point')
    info['PHOTSYS'] = ("SPARC4", 'photometric system')
    info['REFCAT'] = (ref_catalog_name, 'reference catalog key')
    info['TIMEKEY'] = (time_key, 'keyword used to extract times')
    info['TIMEFMT'] = (time_format, 'time format in img files')
    info['TIMESCL'] = (time_scale, 'time scale')
    info['POLAR'] = (polarimetry, 'polarimetry mode')

    #info['PHBAND'] = (hdr["FILTER"], 'photometric band pass')
    #info['WAVELEN'] = (550., 'band pass central wavelength [nm]')
    #info['BANDWIDT'] = (300., 'photometric band width [nm]')

    # generate the photometric time series product
    s4p.photTimeSeriesProduct(tsdata, catalog_names, info=info, filename=output)
 
    return output


def get_waveplate_angle(wppos) :

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

    angles = [0.,22.5,45.,67.5,90.,
             112.5,135.,157.5,180.,
             202.5,225.,247.5,270.,
             292.5,315.,337.5]

    return angles[wppos]


def load_list_of_sci_image_catalogs(sci_list, wppos_key="WPPOS", polarimetry=False) :
    
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

    if polarimetry :
        beam1, beam2 = {}, {}
    
        waveplate_angles = np.array([])
        apertures = np.array([])
    
        for i in range(len(sci_list)) :
    
            hdulist = fits.open(sci_list[i])

            wppos = int(hdulist[0].header[wppos_key])
            waveplate_angles = np.append(waveplate_angles, get_waveplate_angle(wppos-1))

            phot1data, phot2data = [], []
            
            for ext in range(1,len(hdulist)) :
                if (ext % 2) != 0 :
                    if i == 0 :
                        apertures = np.append(apertures,hdulist[ext].data[0][11])
                        nsources = len(hdulist[ext].data)
                        
                    phot1data.append(Table(hdulist[ext].data))
                    phot2data.append(Table(hdulist[ext+1].data))
                else :
                    continue
            beam1[sci_list[i]] = phot1data
            beam2[sci_list[i]] = phot2data

        #return beam1, beam2, waveplate_angles*u.degree, apertures, nsources
        return beam1, beam2, waveplate_angles, apertures, nsources
        
    else :
        beam = {}
    
        apertures = np.array([])

        for i in range(len(sci_list)) :
    
            hdulist = fits.open(sci_list[i])
            
            photdata = []

            for ext in range(1,len(hdulist)) :
                if i == 0 :
                    apertures = np.append(apertures,hdulist[ext].data[0][11])
                    nsources = len(hdulist[ext].data)

                photdata.append(Table(hdulist[ext].data))
                    
            beam[sci_list[i]] = photdata

        return beam, apertures, nsources



def get_qflux(beam, filename, aperture_index, source_index) :
    
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

    Returns
    -------
    qflux : Qfloat
        return flux+/-flux_err information from input catalog
    """
    
    
    umag = ufloat(beam[filename][aperture_index]["MAG"][source_index],
                  beam[filename][aperture_index]["EMAG"][source_index])
    
    uflux = 10**(-0.4*umag)

    qflux = QFloat(uflux.nominal_value, uflux.std_dev)
    
    return qflux
    
    
def run_polarimetry(p, sci_list, pos=[], output="", source_index=None, wave_plate='half_wave', use_perr=False, title_label=None, plot_snr=False, plot_aper=False, plot=False) :

    # define output polarimetry product file name
    output = sci_list[0].replace("_proc.fits","_polar.fits")
        
    # get data from a list of science image products
    beam1, beam2, waveplate_angles, apertures, nsources = load_list_of_sci_image_catalogs(sci_list, polarimetry=True)

    # set initial and final source indexes
    ini_src_idx, end_src_idx = 0, nsources
    
    # if a source index is given, run polarimetry only for this target
    if source_index != None :
        ini_src_idx = source_index
        end_src_idx = source_index + 1

    print("Number of sources in catalog: {}".format(nsources))
    print("Number of apertures: {}  varying from {} to {} in steps of {} pix".format(len(apertures),apertures[0],apertures[-1],np.abs(np.nanmedian(apertures[1:]-apertures[:-1]))))

    if wave_plate == 'half_wave' :
    
        # set number of free parameters in the fit
        number_of_free_params = 4 #??
        
        # initialize polarimetry fitter
        pol = SLSDualBeamPolarimetry('halfwave', compute_k=True, zero=0)
    
        min_merit_figure = 1e10
        best_pol_result = None
        best_aperture = np.nan
        best_wp_angles = deepcopy(waveplate_angles)
        
        # loop over each source in the catalog
        #for j in range(nsources) :
        for j in range(ini_src_idx,end_src_idx) :
            
            err_p, chisqr = [], []
            
            for i in range(len(apertures)):
            
                n_fo, n_fe = [], []
                en_fo, en_fe = [], []

                for filename in sci_list :
                    # get fluxes as qfloats
                    fo = get_qflux(beam1, filename, i, j)
                    fe = get_qflux(beam2, filename, i, j)
                    #print(filename, i, j, fo, fe)
                    n_fo.append(fo.nominal)
                    n_fe.append(fe.nominal)
                    en_fo.append(fo.std_dev)
                    en_fe.append(fe.std_dev)

                n_fo, n_fe = np.array(n_fo), np.array(n_fe)
                en_fo, en_fe = np.array(en_fo), np.array(en_fe)

                if plot_snr :
                    plt.title("SNR for source index: {} and aperture radius: {} pixels".format(j,apertures[i]))
                    plt.plot(waveplate_angles,n_fo/en_fo,label='N-beam')
                    plt.plot(waveplate_angles,n_fe/en_fe,label='S-beam')
                    plt.xlabel("Wave plate angle [deg]")
                    plt.ylabel(r"signal-to-noise ratio")
                    plt.legend()
                    plt.show()

                keep = np.isfinite(n_fo)
                keep &= np.isfinite(n_fe)
                keep &= (n_fo > np.nanmedian(en_fo)) & (n_fe > np.nanmedian(en_fe))
                keep &= (en_fo > 0) & (en_fe > 0)

                #print("Number of exposures removed: {}".format(len(n_fo[~keep])))

                epi, chi2 = np.nan, np.nan
                norm = None
                try :
                    norm = pol.compute(waveplate_angles[keep], n_fo[keep], n_fe[keep], f_ord_error=en_fo[keep], f_ext_error=en_fe[keep])
                    observed_model = halfwave_model(waveplate_angles[keep], norm.q.nominal, norm.u.nominal, zero=None)
                    epi = norm.p.std_dev
                    n, m = len(observed_model), number_of_free_params
                    chi2 = np.nansum(((observed_model - norm.zi.nominal)/norm.zi.std_dev)**2) / (n - m)
                except :
                    print("WARNING: could not calculate polarimetry for source_index={} and aperture={} pixels".format(j,apertures[i]))
                    
                err_p.append(epi)
                chisqr.append(chi2)

                merit_figure = chi2
                if use_perr :
                    merit_figure = epi

                if np.isfinite(merit_figure) and merit_figure < min_merit_figure :
                    best_pol_result = norm
                    min_pol_err, min_pol_chi2 = epi, chi2
                    min_merit_figure = merit_figure
                    best_aperture = apertures[i]
                    best_wp_angles = deepcopy(waveplate_angles[keep])
                    
            if plot_aper :
                if use_perr :
                    plt.plot(apertures, err_p, '-', label="Source {}".format(j))
                    plt.vlines(best_aperture, 0, np.max(err_p), colors="r", linestyles='--', label="Best aperture: {} pix".format(best_aperture))
                    plt.ylabel(r"$\sigma_p$")
                else :
                    plt.plot(apertures, chisqr, '-', label="Source {}".format(j))
                    plt.vlines(best_aperture, 0, np.max(chisqr), colors="r", linestyles='--', label="Best aperture: {} pix".format(best_aperture))
                    plt.ylabel(r"$\chi^2$")
                plt.xlabel("Aperture radius [pixel]")
                plt.legend()
                plt.show()
            
            # calculate residuals
            resids = halfwave_model(best_wp_angles, best_pol_result.q.nominal, best_pol_result.u.nominal) - best_pol_result.zi.nominal

            # calculate rms of residuals
            sig_res = np.nanstd(resids)
            n, m = len(resids), number_of_free_params
            chi2 = np.nansum((resids/best_pol_result.zi.std_dev)**2) / (n - m)
    
            # print results
            print("Best aperture radius: {} pixels".format(best_aperture))
            print("Polarization in Q: {}".format(best_pol_result.q))
            print("Polarization in U: {}".format(best_pol_result.u))
            print("Total linear polarization p: {}".format(best_pol_result.p))
            print("Angle of polarization theta: {}".format(best_pol_result.theta))
            print("Free constant k: {}".format(best_pol_result.k))
            print("RMS of {} residuals: {:.5f}".format("zi", sig_res))
            print("Reduced chi-square (n={}, DOF={}): {:.2f}".format(n,n-m,chi2))

            if plot :
                # plot best polarimetry results
                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})

                if title_label != "":
                    axes[0].set_title(title_label)
                
                # define grid of position angle points for model
                pos_model = np.arange(0, 360, p['POS_MODEL_SAMPLING'])*u.degree
                # Plot the model
                best_fit_model = halfwave_model(pos_model, best_pol_result.q.nominal, best_pol_result.u.nominal)
                axes[0].plot(pos_model, best_fit_model,'r:', alpha=0.8, label='Best fit model')
                #axes[0].fill_between(pos_model, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3, edgecolor="none")
 
                # Plot data
                axes[0].errorbar(best_wp_angles, best_pol_result.zi.nominal, yerr=best_pol_result.zi.std_dev, fmt='ko', ms=2, capsize=2, lw=0.5, alpha=0.9, label='data')
                axes[0].set_ylabel(r"$Z(\phi) = \frac{f_\parallel(\phi)-f_\perp(\phi)}{f_\parallel(\phi)+f_\perp(\phi)}$", fontsize=16)
                axes[0].legend(fontsize=16)
                axes[0].tick_params(axis='x', labelsize=14)
                axes[0].tick_params(axis='y', labelsize=14)
        
                # Print q, u, p and theta values
                ylims = axes[0].get_ylim()
                qlab = "q: {:.2f}+-{:.2f} %".format(100*best_pol_result.q.nominal,100*best_pol_result.q.std_dev)
                ulab = "u: {:.2f}+-{:.2f} %".format(100*best_pol_result.u.nominal,100*best_pol_result.u.std_dev)
                plab = "p: {:.2f}+-{:.2f} %".format(100*best_pol_result.p.nominal,100*best_pol_result.p.std_dev)
                thetalab = r"$\theta$: {:.2f}+-{:.2f} deg".format(best_pol_result.theta.nominal,best_pol_result.theta.std_dev)
                axes[0].text(-10, ylims[1]-0.06,'{}\n{}\n{}\n{}'.format(qlab, ulab, plab, thetalab), size=12)

                # Plot residuals
                axes[1].errorbar(best_wp_angles, resids, yerr=best_pol_result.zi.std_dev, fmt='ko', alpha=0.5, label='residuals')
                axes[1].set_xlabel(r"waveplate position angle, $\phi$ [deg]", fontsize=16)
                axes[1].hlines(0., 0, 360, color="k", linestyles=":", lw=0.6)
                axes[1].set_ylim(-5*sig_res,+5*sig_res)
                axes[1].set_ylabel(r"residuals", fontsize=16)
                axes[1].tick_params(axis='x', labelsize=14)
                axes[1].tick_params(axis='y', labelsize=14)
                plt.show()

    elif wave_plate == 'quarter_wave' :
        pass
    else :
        print("Error: selected wave_plate not recognized, exiting ...")
        exit()
        
    return output


def get_photometric_data_for_polar_catalog(beam1, beam2, sci_list, aperture_index=8, source_index=0) :

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
    """
    
    i, j = aperture_index, source_index

    ra, dec = np.nan, np.nan
    x1, y1 = np.nan, np.nan
    x2, y2 = np.nan, np.nan

    flux1, flux2 = QFloat(0, 0), QFloat(0, 0)
    fwhms = np.array([])
    
    for k in range(len(sci_list)) :

        if k == 0 :
            ra, dec = beam1[sci_list[0]][i]["RA"][j], beam1[sci_list[0]][i]["DEC"][j]
            x1, y1 = beam1[sci_list[0]][i]["X"][j], beam1[sci_list[0]][i]["Y"][j]
            x2, y2 = beam2[sci_list[0]][i]["X"][j], beam2[sci_list[0]][i]["Y"][j]

        flux1 += get_qflux(beam1, sci_list[k], i, j) / (2 * len(sci_list))
        flux2 += get_qflux(beam2, sci_list[k], i, j) / (2 * len(sci_list))

        fwhms = np.append(fwhms, beam1[sci_list[k]][i]["FWHMX"][j])
        fwhms = np.append(fwhms, beam1[sci_list[k]][i]["FWHMY"][j])
        fwhms = np.append(fwhms, beam2[sci_list[k]][i]["FWHMX"][j])
        fwhms = np.append(fwhms, beam2[sci_list[k]][i]["FWHMY"][j])

    fwhm = np.nanmedian(fwhms)

    flux = flux1 + flux2
    mag = -2.5 * np.log10(flux)

    return ra, dec, x1, y1, x2, y2, mag.nominal, mag.std_dev, fwhm


def compute_polarimetry(sci_list, output_filename="", wppos_key='WPPOS', save_output=True, wave_plate='halfwave', compute_k=True, fit_zero=False, zero=0, base_aperture=8, force=False)  :

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
    if output_filename == "" :
        if wave_plate == 'halfwave' :
            output_filename = sci_list[0].replace("_proc.fits","_l2_polar.fits")
        elif wave_plate == 'quarterwave' :
            output_filename = sci_list[0].replace("_proc.fits","_l4_polar.fits")
        else :
            print("ERROR: wave plate mode not supported, exiting ...")
            exit()

    if os.path.exists(output_filename) and not force :
        print("There is already a polarimetry product :", output_filename)
        return output_filename

    # get data from a list of science image products
    beam1, beam2, waveplate_angles, apertures, nsources = load_list_of_sci_image_catalogs(sci_list, wppos_key=wppos_key, polarimetry=True)

    print("Number of sources in catalog: {}".format(nsources))
    print("Number of apertures: {}  varying from {} to {} in steps of {} pix".format(len(apertures),apertures[0],apertures[-1],np.abs(np.nanmedian(apertures[1:]-apertures[:-1]))))

    # initialize list of catalogs
    polar_catalogs, sources = [], []

    # set number of free parameters in the fit
    number_of_free_params = 4 # can we get this number from pol?

    if wave_plate == 'halfwave' :
        zero = 0
    
    if wave_plate == 'quarterwave' :
        number_of_free_params += 1

    if fit_zero :
        zero = None
        # we do not fit zero and k simultaneously
        compute_k = False

    if compute_k :
        number_of_free_params += 1

    pol = SLSDualBeamPolarimetry(wave_plate, compute_k=compute_k, zero=zero)

    # loop over each source in the catalog
    for j in range(nsources) :

        print("Calculating {} polarimetry for source {} of {}".format(wave_plate,j+1,nsources))

        # create an empty dict to store catalog of current aperture
        catalog = {}

        ra, dec, x1, y1, x2, y2, mag, mag_err, fwhm = get_photometric_data_for_polar_catalog(beam1, beam2, sci_list, aperture_index=base_aperture, source_index=j)

        sources_info = {"RA": ra, "DEC": dec, "X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "MAG": mag, "EMAG": mag_err, "FWHM": fwhm}

        sources.append(sources_info)

        # loop over each aperture
        for i in range(len(apertures)):
        
            n_fo, n_fe = [], []
            en_fo, en_fe = [], []

            for filename in sci_list :
                # get fluxes as qfloats
                fo = get_qflux(beam1, filename, i, j)
                fe = get_qflux(beam2, filename, i, j)
                #print(filename, i, j, fo, fe)
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
            for k in range(len(sci_list)) :
                if keep[k] :
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
            
            norm = pol.compute(waveplate_angles[keep], n_fo[keep], n_fe[keep], f_ord_error=en_fo[keep], f_ext_error=en_fe[keep])

            observed_model = np.full_like(waveplate_angles[keep],np.nan)

            if wave_plate == 'halfwave' :
                observed_model = halfwave_model(waveplate_angles[keep], norm.q.nominal, norm.u.nominal)
                    
            elif wave_plate == 'quarterwave' :
                observed_model = quarterwave_model(waveplate_angles[keep], norm.q.nominal, norm.u.nominal, norm.v.nominal, zero= norm.zero.nominal)
                
            try :
                chi2 = np.nansum(((observed_model - norm.zi.nominal)/norm.zi.std_dev)**2) / (number_of_observations - number_of_free_params)
                    
                zi[keep] = norm.zi.nominal
                zi_err[keep] = norm.zi.std_dev

                polar_flag = 0
                    
                qpol, q_err = norm.q.nominal, norm.q.std_dev
                upol, u_err = norm.u.nominal, norm.u.std_dev
                if wave_plate == 'quarterwave' :
                    vpol, v_err = norm.v.nominal, norm.v.std_dev
                ptot, ptot_err = norm.p.nominal, norm.p.std_dev
                theta, theta_err = norm.theta.nominal, norm.theta.std_dev
                k_factor = norm.k
                zero, zero_err = norm.zero.nominal, norm.zero.std_dev

            except :
                print("WARNING: could not calculate polarimetry for source_index={} and aperture={} pixels".format(j,apertures[i]))

            catalog["{}".format(i)] = [i, apertures[i],
                                         qpol, q_err,
                                         upol, u_err,
                                         vpol, v_err,
                                         ptot, ptot_err,
                                         theta, theta_err,
                                         k_factor, k_factor_err,
                                         zero, zero_err,
                                         number_of_observations, number_of_free_params,
                                         chi2, polar_flag]
            for ii in range(len(zi)) :
                catalog["{}".format(i)].append(zi[ii])
                catalog["{}".format(i)].append(zi_err[ii])

            catalog["{}".format(i)] = tuple(catalog["{}".format(i)])
                
        polar_catalogs.append(catalog)

    if save_output :
        info = {}
            
        hdr_start = fits.getheader(sci_list[0])
        hdr_end = fits.getheader(sci_list[-1])
        if "OBJECT" in hdr_start.keys() :
            info['OBJECT'] = (hdr_start["OBJECT"], 'ID of object of interest')
        if "OBSLAT" in hdr_start.keys() :
            info['OBSLAT'] = (hdr_start["OBSLAT"], '[DEG] observatory latitude (N)')
        if "OBSLONG" in hdr_start.keys() :
            info['OBSLONG'] = (hdr_start["OBSLONG"], '[DEG] observatory longitude (E)')
        if "OBSALT" in hdr_start.keys() :
            info['OBSALT'] = (hdr_start["OBSALT"], '[m] observatory altitude')
        info['TELESCOP'] = ('OPD-PE 1.6m', 'telescope')
        if "INSTRUME" in hdr_start.keys() :
            info['INSTRUME'] = (hdr_start["INSTRUME"], 'instrument')
        if "EQUINOX" in hdr_start.keys() :
            info['EQUINOX'] = (hdr_start["EQUINOX"], 'equinox of celestial coordinate system')
        info['PHZEROP'] = (0., '[mag] photometric zero point')
        info['PHOTSYS'] = ("SPARC4", 'photometric system')
        if "CHANNEL" in hdr_start.keys() :
            info['CHANNEL'] = (hdr_start["CHANNEL"], 'Instrument channel')
        info['POLTYPE'] = (wave_plate, 'polarimetry type l/2 or l/4')

        tstart = Time(hdr_start["BJD"], format='jd', scale='utc')
            
        exptime = hdr_end["EXPTIME"]
        if type(exptime) == str :
            if ',' in exptime :
                exptime = exptime.replace(",",".")
            exptime = float(exptime)
            
        tstop = Time(hdr_end["BJD"]+exptime/(24*60*60), format='jd', scale='utc')
            
        info['TSTART'] = (tstart.jd, 'observation start time in BJD')
        info['TSTOP'] = (tstop.jd, 'observation stop time in BJD')
        info['DATE-OBS'] = (tstart.isot, 'TSTART as UTC calendar date')
        info['DATE-END'] = (tstop.isot, 'TSTOP as UTC calendar date')
        info['NEXPS'] = (len(sci_list), 'number of exposures in sequence')

        for k in range(len(sci_list)) :
            hdr = fits.getheader(sci_list[k])
            info["FILE{:04d}".format(k)] = (os.path.basename(sci_list[k]), 'file name of exposure')
            info["EXPT{:04d}".format(k)] = (exptime, 'exposure time (s)')
            info["BJD{:04d}".format(k)] = (hdr["BJD"], 'start time of exposure (BJD)')
            info["WPPO{:04d}".format(k)] = (hdr[wppos_key], 'WP index position of exposure')
            info["WANG{:04d}".format(k)] = (waveplate_angles[k], 'WP angle of exposure (deg)')

        print("Saving output {} polarimetry product: {}".format(wave_plate, output_filename))
        output_hdul = s4p.polarProduct(sources, polar_catalogs, info=info, filename=output_filename)

    return output_filename



def get_polarimetry_results(filename, source_index=0, aperture_index=None, min_aperture=0, max_aperture=1024, plot=False, verbose=False) :

    """ Pipeline module to compute polarimetry for given polarimetric sequence and
        saves the polarimetry data into a FITS SPARC4 product
    
    Parameters
    ----------
    filename : str
        file path for a polarimetry product
    source_index : int
        source index to select only data for a given source
    aperture_index : int
        aperture index to select a given aperture, if not provided
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

    # add 1 to skip the primary extension
    source_index += 1

    # open polarimetry product FITS file
    hdul = fits.open(filename)
    wave_plate = hdul[0].header['POLTYPE']

    # if an aperture index is not given, then consider the one with minimum chi2
    if aperture_index == None :
        ap_range = (hdul[source_index].data['APER']>min_aperture) & (hdul[source_index].data['APER']<max_aperture)
        pickindex = np.nanargmin(hdul[source_index].data['CHI2'][ap_range])
        aperture = hdul[source_index].data['APER'][ap_range][pickindex]
        aperture_index = np.nanargmin(np.abs(hdul[source_index].data['APER']-aperture))
    else :
        # get aperture radius
        aperture = hdul[source_index].data['APER'][aperture_index]

    # get number of exposure in the polarimetric sequence
    nexps = hdul[0].header['NEXPS']

    # get polarization data and the WP position angles
    zis, zierrs, waveplate_angles = np.arange(nexps)*np.nan, np.arange(nexps)*np.nan, np.arange(nexps)*np.nan
    for ii in range(nexps) :
        zis[ii] = hdul[source_index].data["Z{:04d}".format(ii)][aperture_index]
        zierrs[ii] = hdul[source_index].data["EZ{:04d}".format(ii)][aperture_index]
        waveplate_angles[ii] = hdul[0].header["WANG{:04d}".format(ii)]
       
    # filter out nan data
    keep = (np.isfinite(zis)) & (np.isfinite(zierrs))

    if len(zis[keep]) == 0 :
        print("WARNING: no useful polarization data for Source index: {}  and aperture: {} pix ".format(source_index-1, aperture))
        print("Chi-square: {} ".format(hdul[source_index].data['CHI2'][aperture_index]))
        return aperture_index
    
    # get source magnitude
    mag = QFloat(float(hdul[source_index].header["MAG"]),float(hdul[source_index].header["EMAG"]))
  
    # get source coordinates
    ra, dec = hdul[source_index].header["RA"], hdul[source_index].header["DEC"]

    # get polarimetry results
    qpol = QFloat(hdul[source_index].data['Q'][aperture_index], hdul[source_index].data['EQ'][aperture_index])
    upol = QFloat(hdul[source_index].data['U'][aperture_index], hdul[source_index].data['EU'][aperture_index])
    vpol = QFloat(hdul[source_index].data['V'][aperture_index], hdul[source_index].data['EV'][aperture_index])
    ppol = QFloat(hdul[source_index].data['P'][aperture_index], hdul[source_index].data['EP'][aperture_index])
    theta = QFloat(hdul[source_index].data['THETA'][aperture_index], hdul[source_index].data['ETHETA'][aperture_index])
    kcte = QFloat(hdul[source_index].data['K'][aperture_index], hdul[source_index].data['EK'][aperture_index])
    zero = QFloat(hdul[source_index].data['ZERO'][aperture_index], hdul[source_index].data['EZERO'][aperture_index])
    # cast zi data into QFloat
    zi = QFloat(zis[keep], zierrs[keep])

    # calculate polarimetry model and get statistical quantities
    observed_model = np.full_like(waveplate_angles[keep],np.nan)
    if wave_plate == "halfwave" :
        observed_model = halfwave_model(waveplate_angles[keep], qpol.nominal, upol.nominal)
    elif wave_plate == "quarterwave" :
        observed_model = quarterwave_model(waveplate_angles[keep], qpol.nominal, upol.nominal, vpol.nominal, zero=zero.nominal)
        
    n, m = hdul[source_index].data['NOBS'][aperture_index], hdul[source_index].data['NPAR'][aperture_index]
    resids = observed_model - zi.nominal
    sig_res = np.nanstd(resids)
    chi2 = np.nansum((resids/zi.std_dev)**2) / (n - m)

    #print(waveplate_angles[keep], zi, qpol, upol, ppol, theta, kcte)

    # print results
    if verbose :
        print("Source index: i={} ".format(source_index-1))
        print("Source RA={} Dec={} mag={}".format(ra, dec, mag))
        print("Best aperture radius: {} pixels".format(aperture))
        print("Polarization in Q: {}".format(qpol))
        print("Polarization in U: {}".format(upol))
        print("Polarization in V: {}".format(vpol))
        print("Total linear polarization p: {}".format(ppol))
        print("Angle of polarization theta: {}".format(theta))
        print("Free constant k: {}".format(kcte))
        print("Zero of polarization: {}".format(zero))
        print("RMS of {} residuals: {:.5f}".format("zi", sig_res))
        print("Reduced chi-square (n={}, DOF={}): {:.2f}".format(n,n-m,chi2))

    loc["APERTURE_INDEX"] = aperture_index
    loc["APERTURE_RADIUS"] = aperture
    loc["NEXPS"] = nexps
    loc["WAVEPLATE_ANGLES"] = waveplate_angles[keep]
    loc["ZI"] = zi
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
    if plot :
        # set title to appear in the plot header
        title_label = r"Source index: {}    aperture: {} pix    $\chi^2$: {:.2f}    RMS: {:.4f}".format(source_index-1, aperture, chi2, sig_res)

        s4plt.plot_polarimetry_results(loc, title_label=title_label, wave_plate=wave_plate)
    
    hdul.close()

    return loc


def psf_analysis(filename, aperture=10, half_windowsize=15, nsources=0, percentile=99.5, polarimetry=False, plot=False, verbose=False) :

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
    wcs_obj = WCS(hdulist[0].header,naxis=2)
    pixel_scale = proj_plane_pixel_scales(wcs_obj)
    pixel_scale *= 3600
    print("Pixel scale: x: {:.3f} arcsec/pix y: {:.3f} arcsec/pix".format(pixel_scale[0],pixel_scale[1]))
    
    if polarimetry :
    
        catN_label = "CATALOG_POL_N_AP{:03d}".format(aperture)
        catS_label = "CATALOG_POL_S_AP{:03d}".format(aperture)
        photN_data = Table(hdulist[catN_label].data)
        photS_data = Table(hdulist[catS_label].data)
        
        nsources = len(hdulist[catN_label].data)

        fwhms = np.array([])
        
        for j in range(nsources) :
            xN, yN = photN_data["X"][j], photN_data["Y"][j]
            xS, yS = photS_data["X"][j], photS_data["Y"][j]

            fwhms = np.append(fwhms, photN_data["FWHMX"][j])
            fwhms = np.append(fwhms, photN_data["FWHMY"][j])
            fwhms = np.append(fwhms, photS_data["FWHMX"][j])
            fwhms = np.append(fwhms, photS_data["FWHMY"][j])
            
    else :
    
        cat_label = "CATALOG_PHOT_AP{:03d}".format(aperture)
        phot_data = Table(hdulist[cat_label].data)

        if nsources == 0 or nsources > len(hdulist[cat_label].data) :
            nsources = len(hdulist[cat_label].data)

        fwhms = np.array([])

        boxes = np.zeros([nsources,2*half_windowsize+1,2*half_windowsize+1]) * np.nan
        xbox, ybox = None, None
        for j in range(nsources) :
            x, y = phot_data["X"][j], phot_data["Y"][j]
            fwhms = np.append(fwhms, phot_data["FWHMX"][j])
            fwhms = np.append(fwhms, phot_data["FWHMY"][j])
            
            # get  box around source
            box = trim_array(img_data, 2*half_windowsize, (x, y), indices=indices)
            zbox = box[0]

            nx, ny = np.shape(zbox)

            if nx == 2*half_windowsize+1 and ny == 2*half_windowsize+1 :
                maxvalue = np.nanmax(zbox)
                nzbox = zbox/maxvalue
                #print(j, maxvalue, np.nanmedian(zbox))
                boxes[j,:,:] = nzbox
                xbox, ybox = box[1]-x, box[2]-y
                
                #vmin, vmax = np.percentile(nzbox, 1.), np.percentile(nzbox, 99.)
                #s4plt.plot_2d(xbox, ybox, nzbox, LIM=None, LAB=["x (pix)", "y (pix)","flux fraction"], z_lim=[vmin,vmax], title="source: {}".format(j), pfilename="", cmap="gist_heat")


        master_box = np.nanmedian(boxes,axis=0)
        master_box_err = np.nanmedian(np.abs(boxes - master_box),axis=0)
        
        min = np.percentile(master_box, 0.5)
        master_box = master_box - min
        
        max = np.nanmax(master_box)
        master_box /= max
        master_box_err /= max

        #vmin, vmax = np.percentile(master_box, 3.), np.percentile(master_box, 97.)
        #s4plt.plot_2d(xbox*0.33, ybox*0.33, master_box, LIM=None, LAB=["x (arcsec)", "y (arcsec)","flux fraction"], z_lim=[vmin,vmax], title="PSF", pfilename="", cmap="gist_heat")
        
        master_fwhm = _fwhm_loop('gaussian', master_box, xbox, ybox, 0, 0)
        
        print("Median FWHM: {:.3f} pix   Master PSF FWHM: {:.3f} pix".format(np.median(fwhms), master_fwhm))
    
        # multiply x and y by the pixel scale
        xbox *= pixel_scale[0]
        ybox *= pixel_scale[1]
        
        # plot PSF results
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False, gridspec_kw={'hspace': 0.5, 'height_ratios': [1, 1]})
        
        # plot PSF data
        vmin, vmax = np.percentile(master_box, 10), np.percentile(master_box, 99)
        axes[0,0].pcolor(xbox, ybox, master_box, vmin=vmin, vmax=vmax, shading='auto', cmap="cividis")
        axes[0,0].plot(xbox[half_windowsize],np.zeros_like(xbox[half_windowsize]),'--', lw=1., color='white', zorder=3)
        axes[0,0].plot(np.zeros_like(ybox[:,half_windowsize]),ybox[:,half_windowsize],'--', lw=1., color='white', zorder=3)
        axes[0,0].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[0,0].set_ylabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[0,0].set_title("PSF data", pad=10, fontsize=20)
             
        # contour plot
        axes[0,1].contour(xbox, ybox, master_box, vmin=vmin, vmax=vmax, colors='k')
        axes[0,1].plot(xbox[half_windowsize],np.zeros_like(xbox[half_windowsize]),'--', lw=1., color='k', zorder=3)
        axes[0,1].plot(np.zeros_like(ybox[:,half_windowsize]),ybox[:,half_windowsize],'--', lw=1., color='k', zorder=3)
        axes[0,1].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[0,1].set_ylabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[0,1].set_title("PSF contour", pad=10, fontsize=20)
     
        # Fit the data using a Gaussian
        model = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
        fitter = fitting.LevMarLSQFitter()
        best_fit = fitter(model, xbox[half_windowsize], master_box[half_windowsize], weights=1.0/master_box_err[half_windowsize]**2)
        #print(best_fit)
               
        # plot x profile
        axes[1,0].set_title("E-W profile \nFWHM: {:.2f} arcsec".format(2.355*best_fit.stddev.value), pad=10, fontsize=20)
        axes[1,0].errorbar(xbox[half_windowsize],master_box[half_windowsize],yerr=master_box_err[half_windowsize],lw=0.5,fmt='o',color='k')
        axes[1,0].plot(xbox[half_windowsize],best_fit(xbox[half_windowsize]),'-',lw=2,color='brown')
        axes[1,0].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[1,0].set_ylabel("flux", fontsize=16)
        #axes[1,0].legend(fontsize=16)

        best_fit = fitter(model, ybox[:,half_windowsize], master_box[:,half_windowsize], weights=1.0/master_box_err[:,half_windowsize]**2)
        #print(best_fit)
        # plot y profile
        axes[1,1].set_title("N-S profile \nFWHM: {:.2f} arcsec".format(2.355*best_fit.stddev.value), pad=10, fontsize=20)
        axes[1,1].errorbar(ybox[:,half_windowsize],master_box[:,half_windowsize],master_box_err[:,half_windowsize],lw=0.5,fmt='o',color='k')
        axes[1,1].plot(ybox[:,half_windowsize],best_fit(ybox[:,half_windowsize]),'-',lw=2, color='darkblue')
        axes[1,1].set_xlabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[1,1].set_ylabel("flux", fontsize=16)
        #axes[1,1].legend(fontsize=16)
    

        plt.show()
        
        #vmin, vmax = np.percentile(img, 3.), np.percentile(img, 97)
        #s4plt.plot_2d(xw, yw, img, LIM=None, LAB=["x (pix)", "y (pix)","flux fraction"], z_lim=[vmin,vmax], title="PSF plot", pfilename="", cmap="gist_heat")
    
    return loc


def stack_and_reduce_sci_images(p, sci_list, reduce_dir, ref_img="", stack_suffix="", force=True, match_frames=True, polarimetry=False, verbose=False, plot=False) :


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
    if 'OBJECT_REDUCED_IMAGES' in p.keys() :
        del p['OBJECT_REDUCED_IMAGES']
        
    # calculate stack
    p = stack_science_images(p,
                            sci_list,
                            reduce_dir=reduce_dir,
                            force=force,
                            stack_suffix=stack_suffix,
                            polarimetry=polarimetry)
                            
    # set numbe of science reduction loops to avoid memory issues.
    nloops = int(np.ceil(len(sci_list) / p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP']))
    
    # set reference image
    if ref_img == "" :
        ref_img = p['REFERENCE_IMAGE']
        
    for loop in range(nloops) :
        first = p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] * loop
        last = p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] * (loop+1)
        if last > len(sci_list) :
            last = len(sci_list)
            
        if verbose :
            print("Running loop {} of {} -> images in loop: {} to {} ... ".format(loop,nloops,first,last))

        # reduce science data and calculate stack
        p = reduce_science_images(p,
                                  sci_list[first:last],
                                  reduce_dir=reduce_dir,
                                  ref_img=ref_img,
                                  force=force,
                                  match_frames=match_frames,
                                  polarimetry=polarimetry)

    # reduce science data and calculate stack
    #p = s4pipelib.old_reduce_science_images(p, p['objsInPolarL2data'][j][object], data_dir=data_dir, reduce_dir=reduce_dir, force=options.force, match_frames=match_frames, stack_suffix=stack_suffix, polarimetry=True)
    if match_frames and plot :
        if polarimetry :
            s4plt.plot_sci_polar_frame(p['OBJECT_STACK'], percentile=99.5)
        else :
            s4plt.plot_sci_frame(p['OBJECT_STACK'], nstars=20, use_sky_coords=True)

    return p


def polar_time_series(sci_pol_list,
                    reduce_dir="./",
                    ts_suffix="",
                    aperture_index=None,
                    min_aperture=0,
                    max_aperture=1024,
                    force=True) :
    """ Pipeline module to calculate photometry differential time series for a given list of sparc4 sci image products
    
    Parameters
    ----------
    sci_pol_list : list
        list of paths to science polar products
    ts_suffix : str (optional)
        time series suffix to add into the output file name
    reduce_dir : str (optional)
        path to the reduce directory
    aperture_index : int (optional)
        index to select aperture in catalog. Default is None and it will calculate best aperture
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
    output = os.path.join(reduce_dir, "{}_polar_ts.fits".format(ts_suffix))
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

    catalog_names=[]
    ra, dec, src_idx = [], [], []
    # get catalog names in the first sequence. Each catalog correspond to one source
    for hdu in basehdul :
        if hdu.name != "PRIMARY" :
            catalog_names.append(hdu.name)
            ra.append(hdu.header["RA"])
            dec.append(hdu.header["DEC"])
            src_idx.append(hdu.header["SRCINDEX"])
            
            # initialize time series vector for each quantity
            tsdata[hdu.name] = {}
            
            tsdata[hdu.name]['TIME'] = np.array([])
            tsdata[hdu.name]['MAG'] = np.array([])
            tsdata[hdu.name]['EMAG'] = np.array([])
            tsdata[hdu.name]['FWHM'] = np.array([])
            tsdata[hdu.name]['X1'] = np.array([])
            tsdata[hdu.name]['Y1'] = np.array([])
            tsdata[hdu.name]['X2'] = np.array([])
            tsdata[hdu.name]['Y2'] = np.array([])
            
            tsdata[hdu.name]['Q'] = np.array([])
            tsdata[hdu.name]['EQ'] = np.array([])
            tsdata[hdu.name]['U'] = np.array([])
            tsdata[hdu.name]['EU'] = np.array([])
            tsdata[hdu.name]['V'] = np.array([])
            tsdata[hdu.name]['EV'] = np.array([])
            tsdata[hdu.name]['P'] = np.array([])
            tsdata[hdu.name]['EP'] = np.array([])
            tsdata[hdu.name]['THETA'] = np.array([])
            tsdata[hdu.name]['ETHETA'] = np.array([])
            tsdata[hdu.name]['K'] = np.array([])
            tsdata[hdu.name]['EK'] = np.array([])
            tsdata[hdu.name]['ZERO'] = np.array([])
            tsdata[hdu.name]['EZERO'] = np.array([])
            tsdata[hdu.name]['NOBS'] = np.array([])
            tsdata[hdu.name]['NPAR'] = np.array([])
            tsdata[hdu.name]['CHI2'] = np.array([])

    ti, tf = 0, 0

    for i in range(len(sci_pol_list)) :
    
        hdul = fits.open(sci_pol_list[i])
        header = hdul[0].header

        mid_bjd = (header["TSTART"] + header["TSTOP"]) / 2
        
        if i == 0 :
            ti = header["TSTART"]
        if i == len(sci_pol_list) - 1 :
            tf = header["TSTOP"]

        for key in catalog_names :
            
            hdr = hdul[key].header

            tsdata[key]['TIME'] = np.append(tsdata[key]['TIME'],mid_bjd)
            tsdata[key]['MAG'] = np.append(tsdata[key]['MAG'],hdr['MAG'])
            tsdata[key]['EMAG'] = np.append(tsdata[key]['EMAG'],hdr['EMAG'])
            tsdata[key]['FWHM'] = np.append(tsdata[key]['FWHM'],hdr['FWHM'])
            tsdata[key]['X1'] = np.append(tsdata[key]['X1'],hdr['X1'])
            tsdata[key]['Y1'] = np.append(tsdata[key]['Y1'] ,hdr['Y1'])
            tsdata[key]['X2'] = np.append(tsdata[key]['X2'],hdr['X2'])
            tsdata[key]['Y2'] = np.append(tsdata[key]['Y2'],hdr['Y2'])

            # read polarimetry results for the base sequence in the time series
            polar = get_polarimetry_results(sci_pol_list[i],
                                        source_index=hdr["SRCINDEX"],
                                        aperture_index=aperture_index,
                                        min_aperture=min_aperture,
                                        max_aperture=max_aperture)

            tsdata[key]['Q'] = np.append(tsdata[key]['Q'], polar['Q'].nominal)
            tsdata[key]['EQ'] = np.append(tsdata[key]['EQ'], polar['Q'].std_dev)
            tsdata[key]['U'] = np.append(tsdata[key]['U'], polar['U'].nominal)
            tsdata[key]['EU'] = np.append(tsdata[key]['EU'], polar['U'].std_dev)
            tsdata[key]['V'] = np.append(tsdata[key]['V'], polar['V'].nominal)
            tsdata[key]['EV'] = np.append(tsdata[key]['EV'], polar['V'].std_dev)
            tsdata[key]['P'] = np.append(tsdata[key]['P'], polar['P'].nominal)
            tsdata[key]['EP'] = np.append(tsdata[key]['EP'], polar['P'].std_dev)
            tsdata[key]['THETA'] = np.append(tsdata[key]['THETA'], polar['THETA'].nominal)
            tsdata[key]['ETHETA'] = np.append(tsdata[key]['ETHETA'], polar['THETA'].std_dev)
            tsdata[key]['K'] = np.append(tsdata[key]['K'], polar['K'].nominal)
            tsdata[key]['EK'] = np.append(tsdata[key]['EK'], polar['K'].std_dev)
            tsdata[key]['ZERO'] = np.append(tsdata[key]['ZERO'], polar['ZERO'].nominal)
            tsdata[key]['EZERO'] = np.append(tsdata[key]['EZERO'] , polar['ZERO'].std_dev)
            tsdata[key]['NOBS'] = np.append(tsdata[key]['NOBS'], polar['NOBS'])
            tsdata[key]['NPAR'] = np.append(tsdata[key]['NPAR'], polar['NPAR'])
            tsdata[key]['CHI2'] = np.append(tsdata[key]['CHI2'], polar['CHI2'])

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
    info['APINDX'] = (aperture_index, 'selected aperture index')
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
    s4p.polarTimeSeriesProduct(tsdata, catalog_names, info=info, filename=output)
 
    return output
