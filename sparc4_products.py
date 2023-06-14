"""
    Created on May 2 2022

    Description: Library for the SPARC4 pipeline products

    @author: Eder Martioli <martioli@iap.fr>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys
import numpy as np
import astropy.io.fits as fits
from typing import Collection, Union
from astropy import units as u
from astropop.framedata import FrameData

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from uncertainties import ufloat, umath

import sparc4_utils as s4utils

from copy import deepcopy

from astropy.table import Table

ExtensionHDU = Union[fits.ImageHDU, fits.BinTableHDU]
HDU = Union[fits.PrimaryHDU, ExtensionHDU]

def create_hdu_list(hdus: Collection[HDU]) -> fits.HDUList:
    """
    Takes a collection of fits HDUs and converts into an HDUList ready to be saved to a file
    :param hdus: The collection of fits HDUs
    :return: A fits HDUList ready to be saved to a file
    """
    hdu_list = fits.HDUList()
    for hdu in hdus:
        if len(hdu_list) == 0:
            hdu.header['NEXTEND'] = len(hdus) - 1
        else:
            hdu.header.remove('NEXTEND', ignore_missing=True)
        hdu_list.append(hdu)
    return hdu_list



def masterCalibration(list_of_imgs, img_data=[], err_data=[], mask_data=[], info={}, filename="") :
    """ Create a master calibration FITS image product (BIAS, DARK, FLAT, etc)

    Parameters
    ----------
    list_of_imgs : list
        list of file paths for the original images used to calculate master
    img_data : numpy.ndarray (n x m)
        float array containing the master calibration data
    err_data : numpy.ndarray (n x m)
        float array containing the uncertainty of master calibration data
    mask_data : numpy.ndarray (n x m)
        uint array containing the mask data
    info : dict
        dictionary with additional header cards to include in the header of product
        The following format must be used:
            info = {key1: (value1, comment1), key2: (value2, comment2), ... }
    filename : str, optional
        The output file name to save product. If empty, file won't be saved.

    Returns
    -------
    hdu_list : astropy.io.fits.HDUList
        output hdu_list top-level FITS object.
    """

    # get header from base image
    baseheader = fits.getheader(list_of_imgs[0])

    # add information about data in the product
    info['DATA0'] = ('IMG DATA', 'content of slice 0 in cube')
    info['DATA1'] = ('ERR DATA', 'content of slice 1 in cube')
    info['DATA2'] = ('MASK DATA', 'content of slice 2 in cube')

    # get number of images in the list
    ninimgs = len(list_of_imgs)

    # add number of images information
    info['NINIMGS'] = (ninimgs, 'number of input images')

    # loop over each image name in the list
    for i in range(ninimgs) :
        # get file basename and add it to the info dict
        basename = os.path.basename(list_of_imgs[i])
        info['IN{:06d}'.format(i)] = (basename, 'input file {} of {}'.format(i,ninimgs))

    # create primary hdu with header of base image
    primary_hdu = fits.PrimaryHDU(header=baseheader)

    # add keys given by the info dict
    for key in info.keys() :
        primary_hdu.header.set(key, info[key][0], info[key][1])

    # define default arrays in case they are not provided
    if len(img_data) == 0:
        img_data = np.empty((1024,1024), dtype=float) * np.nan
    if len(err_data) == 0:
        err_data = np.full_like(img_data,np.nan)
    if len(mask_data) == 0:
        mask_data = np.zeros_like(img_data)

    # set data cube into primary extension
    primary_hdu.data = np.array([img_data,err_data,mask_data])

    # create hdu list
    hdu_list = create_hdu_list([primary_hdu])

    # write FITS file if filename is given
    if filename != "" :
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def getFrameFromMasterCalibration(filename) :

    hdu_list = fits.open(filename)

    img_data = hdu_list[0].data[0]
    err_data = hdu_list[0].data[1]
    mask_data = hdu_list[0].data[2]

    header = hdu_list[0].header

    unit=None
    if header['BUNIT'] == 'electron' :
        unit=u.electron

    frame = FrameData(data=img_data, unit=unit, uncertainty=err_data, mask=mask_data, header=header)

    return frame



def scienceImageProduct(original_image, img_data=[], err_data=[], mask_data=[], info={}, catalogs=[], polarimetry=False,  skip_ref_catalogs=True, filename="", catalog_beam_ids=["S","N"], wcs_header=None, time_key="DATE-OBS", ra="", dec="") :
    """ Create a Science FITS image product

    Parameters
    ----------
    original_image : str
        file path for the original image
    img_data : numpy.ndarray (n x m)
        float array containing the master calibration data
    err_data : numpy.ndarray (n x m)
        float array containing the uncertainty of master calibration data
    mask_data : numpy.ndarray (n x m)
        uint array containing the mask data
    info : dict
        dictionary with additional header cards to include in the header of product
        The following format must be used:
            info = {key1: (value1, comment1), key2: (value2, comment2), ... }
    catalogs : list of dict catalogs
        a catalog is a dictionary with information about detected sources
        The following format must be used for each catalog in the list:
            catalog = {star_id1 : (1, ra1, dec1, x1, y1, fwhm_x1, fwhm_y1, mag1, emag1, mag_sky1, emag_sky1, aper1, flag1),
                       star_id2 : (2, ra2, dec2, x2, y2, fwhm_x2, fwhm_y2, mag2, emag2, mag_sky2, emag_sky2, aper2, flag2),
                       .... }
            flag (int):
                0 : single star, all aperture pixels used, no issues in the photometry
                1 : single star, part of pixels in aperture have been rejected, no issues in the photometry
                2 : single star, issues in the photometry
                3 : blended star, all aperture pixels used, no issues in the photometry
                4 : blended star, part of pixels in aperture have been rejected, no issues in the photometry
                5 : blended star, issues in the photometry
                6 : not a star, all aperture pixels used, no issues in the photometry
                7 : not a star, part of pixels in aperture have been rejected, no issues in the photometry
                8 : not a star, issues in the photometry
    polarimetry : bool
        Boolean to set polarimetry product
    skip_ref_catalogs : bool
        Boolean to skip the first catalog in the list (for photometry) or the first two catalogs in the list (for polarimetry).
    filename : str, optional
        The output file name to save product. If empty, file won't be saved.
    catalog_beam_ids : list
        List of two strings to label the two polarimetric beams. Deafult is ["S", "N"], for South and North beams.
    wcs_header : FITS header
        FITS header containing the WCS information. This will be appended to the main header.
    time_key : str, optional
        string to point to the main date keyword in FITS header
    ra : str, optional
        string to overwrite header RA (Right Ascension) keyword
    dec : str, optional
        string to overwrite header DEC (Declination) keyword

    Returns
    -------
    hdu_list : astropy.io.fits.HDUList
        output hdu_list top-level FITS object.
    """

    # get header from base image
    baseheader = fits.getheader(original_image)

    # add information about data in the product
    info['DATA0'] = ('IMG DATA', 'content of slice 0 in cube')
    info['DATA1'] = ('ERR DATA', 'content of slice 1 in cube')
    info['DATA2'] = ('MASK DATA', 'content of slice 2 in cube')

    # get file basename and add it to the info dict
    basename = os.path.basename(original_image)
    info['ORIGIMG'] = (basename, 'original file name')
    info['POLAR'] = (polarimetry, 'polarimetry frame')

    if wcs_header :
        baseheader += wcs_header

    baseheader = s4utils.set_timecoords_keys(baseheader, time_key=time_key, ra=ra, dec=dec)

    # create primary hdu with header of base image
    primary_hdu = fits.PrimaryHDU(header=baseheader)

    # add keys given by the info dict
    for key in info.keys() :
        primary_hdu.header.set(key, info[key][0], info[key][1])

    # define default arrays in case they are not provided
    if len(img_data) == 0:
        img_data = np.empty((1024,1024), dtype=float) * np.nan
    if len(err_data) == 0:
        err_data = np.full_like(img_data,np.nan)
    if len(mask_data) == 0:
        mask_data = np.zeros_like(img_data)

    # set data cube into primary extension
    primary_hdu.data = np.array([img_data,err_data,mask_data])

    hdus = [primary_hdu]

    ini_catalog = 0
    if skip_ref_catalogs and polarimetry :
        ini_catalog = 2
    elif skip_ref_catalogs :
        ini_catalog = 1

    for j in range(ini_catalog,len(catalogs)) :
        # create empty header for catalog extension
        catalog_header = fits.PrimaryHDU().header

        # add number of objects in catalog table
        catalog_header.set("NOBJCAT", len(catalogs[j].keys()), "Number of objects in the catalog")

        # collect catalog data
        catdata = []
        for key in catalogs[j].keys() :
            catdata.append(catalogs[j][key])

        # set names and data format for each column in the catalog table
        dtype=[('INDEX', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('X', 'f8'), ('Y', 'f8'), ('FWHMX', 'f8'), ('FWHMY', 'f8'), ('MAG', 'f8'), ('EMAG', 'f8'), ('SKYMAG', 'f8'), ('ESKYMAG', 'f8'), ('APER', 'i4'), ('FLAG', 'i4')]

        # cast catalog data into numpy array
        catalog_array = np.array(catdata, dtype=dtype)

        # get photometry aperture value for the catalog label
        aperture_value = catalog_array[0][11]

        cat_label = "UNLABELED_CATALOG"

        if polarimetry :
            if j == 0 :
                catalog_header.set("POLBEAM", catalog_beam_ids[0], "Polar beam: [N]orth or [S]outh")
                cat_label = "REF_CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[0],aperture_value)
                pass
            elif j == 1 :
                catalog_header.set("POLBEAM", catalog_beam_ids[1], "Polar beam: [N]orth or [S]outh")
                cat_label = "REF_CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[1],aperture_value)
                pass
            else :
                if (j % 2) == 0 :
                    catalog_header.set("POLBEAM", catalog_beam_ids[0], "Polar beam: [N]orth or [S]outh")
                    #cat_label = "CATALOG_POL_{}_{:04d}".format(catalog_beam_ids[0],int(j/2-1))
                    cat_label = "CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[0],aperture_value)
                else :
                    catalog_header.set("POLBEAM", catalog_beam_ids[1], "Polar beam: [N]orth or [S]outh")
                    #cat_label = "CATALOG_POL_{}_{:04d}".format(catalog_beam_ids[1],int((j-1)/2-1))
                    cat_label = "CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[1],aperture_value)
        else :
            if j == 0 :
                cat_label = "REF_CATALOG_PHOT_AP{:03d}".format(aperture_value)
                pass
            else :
                #cat_label = "CATALOG_PHOT_{:04d}".format(j-1)
                cat_label = "CATALOG_PHOT_AP{:03d}".format(aperture_value)

        hdu_catalog = fits.TableHDU(data=catalog_array, header=catalog_header, name=cat_label)

        # set each column unit
        column_units = ["", "DEG", "DEG", "PIXEL", "PIXEL", "PIXEL", "PIXEL", "MAG", "MAG", "MAG", "MAG", "PIXEL", ""]

        # add description for each column in the header
        for i in range(len(column_units)) :
            if column_units[i] != "" :
                hdu_catalog.header.comments["TTYPE{:d}".format(i+1)] = "units of {}".format(column_units[i])

        # append catalog hdu to hdulist
        hdus.append(hdu_catalog)

    # create hdu list
    hdu_list = create_hdu_list(hdus)

    # write FITS file if filename is given
    if filename != "" :
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def readScienceImagCatalogs(input) :
    """ Pipeline module to read catalogs in a sci image product.

    Parameters
    ----------
    input : str
        input image path

    Returns
    -------
    catalogs: list of dicts
        returns a list of catalog dictionaries
    """
    hdu_list = fits.open(input)

    catalogs = []

    for hdu in hdu_list :
        if hdu.name != 'PRIMARY' :
            catdata = hdu.data
            apercat = {}
            for i in range(len(catdata)) :
                apercat["{}".format(i)] = catdata[i]
            catalogs.append(apercat)

    return catalogs


def scienceImageLightProduct(original_image, img_data=[], info={}, catalogs=[], polarimetry=False,  skip_ref_catalogs=True, filename="", catalog_beam_ids=["S","N"], wcs_header=None, time_key="DATE-OBS", ra="", dec="") :
    """ Create a Science FITS image product

    Parameters
    ----------
    original_image : str
        file path for the original image
    img_data : numpy.ndarray (n x m)
        float array containing the master calibration data
    info : dict
        dictionary with additional header cards to include in the header of product
        The following format must be used:
            info = {key1: (value1, comment1), key2: (value2, comment2), ... }
    catalogs : list of dict catalogs
        a catalog is a dictionary with information about detected sources
        The following format must be used for each catalog in the list:
            catalog = {star_id1 : (1, ra1, dec1, x1, y1, fwhm_x1, fwhm_y1, mag1, emag1, mag_sky1, emag_sky1, aper1, flag1),
                       star_id2 : (2, ra2, dec2, x2, y2, fwhm_x2, fwhm_y2, mag2, emag2, mag_sky2, emag_sky2, aper2, flag2),
                       .... }
            flag (int):
                0 : single star, all aperture pixels used, no issues in the photometry
                1 : single star, part of pixels in aperture have been rejected, no issues in the photometry
                2 : single star, issues in the photometry
                3 : blended star, all aperture pixels used, no issues in the photometry
                4 : blended star, part of pixels in aperture have been rejected, no issues in the photometry
                5 : blended star, issues in the photometry
                6 : not a star, all aperture pixels used, no issues in the photometry
                7 : not a star, part of pixels in aperture have been rejected, no issues in the photometry
                8 : not a star, issues in the photometry
    polarimetry : bool
        Boolean to set polarimetry product
    skip_ref_catalogs : bool
        Boolean to skip the first catalog in the list (for photometry) or the first two catalogs in the list (for polarimetry).
    filename : str, optional
        The output file name to save product. If empty, file won't be saved.
    catalog_beam_ids : list
        List of two strings to label the two polarimetric beams. Deafult is ["S", "N"], for South and North beams.
    wcs_header : FITS header
        FITS header containing the WCS information. This will be appended to the main header.
    time_key : str, optional
        string to point to the main date keyword in FITS header
    ra : str, optional
        string to overwrite header RA (Right Ascension) keyword
    dec : str, optional
        string to overwrite header DEC (Declination) keyword

    Returns
    -------
    hdu_list : astropy.io.fits.HDUList
        output hdu_list top-level FITS object.
    """

    # get header from base image
    baseheader = fits.getheader(original_image)

    # add information about data in the product
    info['DATA0'] = ('IMG DATA', 'content of slice 0 in cube')

    # get file basename and add it to the info dict
    basename = os.path.basename(original_image)
    info['ORIGIMG'] = (basename, 'original file name')
    info['POLAR'] = (polarimetry, 'polarimetry frame')

    if wcs_header :
        baseheader += wcs_header

    baseheader = s4utils.set_timecoords_keys(baseheader, time_key=time_key, ra=ra, dec=dec)

    # create primary hdu with header of base image
    primary_hdu = fits.PrimaryHDU(header=baseheader)

    # add keys given by the info dict
    for key in info.keys() :
        primary_hdu.header.set(key, info[key][0], info[key][1])

    # define default arrays in case they are not provided
    if len(img_data) == 0:
        img_data = np.empty((1024,1024), dtype=float) * np.nan

    # set data cube into primary extension
    primary_hdu.data = np.array(img_data)

    hdus = [primary_hdu]

    ini_catalog = 0
    if skip_ref_catalogs and polarimetry :
        ini_catalog = 2
    elif skip_ref_catalogs :
        ini_catalog = 1

    for j in range(ini_catalog,len(catalogs)) :
        # create empty header for catalog extension
        catalog_header = fits.PrimaryHDU().header

        # add number of objects in catalog table
        catalog_header.set("NOBJCAT", len(catalogs[j].keys()), "Number of objects in the catalog")

        # collect catalog data
        catdata = []
        for key in catalogs[j].keys() :
            catdata.append(catalogs[j][key])

        # set names and data format for each column in the catalog table
        dtype=[('INDEX', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('X', 'f8'), ('Y', 'f8'), ('FWHMX', 'f8'), ('FWHMY', 'f8'), ('MAG', 'f8'), ('EMAG', 'f8'), ('SKYMAG', 'f8'), ('ESKYMAG', 'f8'), ('APER', 'i4'), ('FLAG', 'i4')]

        # cast catalog data into numpy array
        catalog_array = np.array(catdata, dtype=dtype)

        # get photometry aperture value for the catalog label
        aperture_value = catalog_array[0][11]

        cat_label = "UNLABELED_CATALOG"

        if polarimetry :
            if j == 0 :
                catalog_header.set("POLBEAM", catalog_beam_ids[0], "Polar beam: [N]orth or [S]outh")
                cat_label = "REF_CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[0],aperture_value)
                pass
            elif j == 1 :
                catalog_header.set("POLBEAM", catalog_beam_ids[1], "Polar beam: [N]orth or [S]outh")
                cat_label = "REF_CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[1],aperture_value)
                pass
            else :
                if (j % 2) == 0 :
                    catalog_header.set("POLBEAM", catalog_beam_ids[0], "Polar beam: [N]orth or [S]outh")
                    #cat_label = "CATALOG_POL_{}_{:04d}".format(catalog_beam_ids[0],int(j/2-1))
                    cat_label = "CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[0],aperture_value)
                else :
                    catalog_header.set("POLBEAM", catalog_beam_ids[1], "Polar beam: [N]orth or [S]outh")
                    #cat_label = "CATALOG_POL_{}_{:04d}".format(catalog_beam_ids[1],int((j-1)/2-1))
                    cat_label = "CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[1],aperture_value)
        else :
            if j == 0 :
                cat_label = "REF_CATALOG_PHOT_AP{:03d}".format(aperture_value)
                pass
            else :
                #cat_label = "CATALOG_PHOT_{:04d}".format(j-1)
                cat_label = "CATALOG_PHOT_AP{:03d}".format(aperture_value)

        hdu_catalog = fits.TableHDU(data=catalog_array, header=catalog_header, name=cat_label)

        # set each column unit
        column_units = ["", "DEG", "DEG", "PIXEL", "PIXEL", "PIXEL", "PIXEL", "MAG", "MAG", "MAG", "MAG", "PIXEL", ""]

        # add description for each column in the header
        for i in range(len(column_units)) :
            if column_units[i] != "" :
                hdu_catalog.header.comments["TTYPE{:d}".format(i+1)] = "units of {}".format(column_units[i])

        # append catalog hdu to hdulist
        hdus.append(hdu_catalog)

    # create hdu list
    hdu_list = create_hdu_list(hdus)

    # write FITS file if filename is given
    if filename != "" :
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def packTimeSeriesData(times, vars=[], labs=[]) :
    """ Pack time series data into a numpy array ready to be saved into a FITS Table HDU

    Parameters
    ----------
    times : numpy.ndarray (N)
        float array containing the N times in BJD
    vars : list of numpy.ndarray (N x M)
        list of float arrays, each containing the N points in the time series for M sources
    xlabel : list of str
        list of strings for the ID name of input variables

    Returns
    -------
    tsarray : numpy.recarray (N x (2 x M + 1))
        output tsarray time series array
    """

    # get number of sources
    nsources = np.shape(vars[0])[1]

    # set array with times
    tsarray = []

    # set names and data format for each column in the catalog table
    dtype=[('TIME', 'f8')]

    for j in range(nsources) :
        for k in range(len(labs)) :
            dtype.append(('{}{:08d}'.format(labs[k],j), 'f8'))

    for i in range(len(times)) :
        visit = [times[i]]
        for j in range(nsources) :
            for k in range(len(labs)) :
                visit.append(vars[k][i][j])
        tsarray.append(tuple(visit))

    # cast coordinates data into numpy array
    tsarray = np.array(tsarray, dtype=dtype)

    return tsarray


def photTimeSeriesProduct(tsdata, catalog_names, info={}, filename="") :
    """ Create a photometric time series FITS product

    Parameters
    ----------
    tsdata : dict
        time series data container
    catalog_names : list
        list of str with catalog names
    info : dict
        dictionary with additional header cards to include in the header of product
        The following format must be used:
            info = {key1: (value1, comment1), key2: (value2, comment2), ... }

    filename : str, optional
        The output file name to save product. If empty, file won't be saved.

    Returns
    -------
    hdu_list : astropy.io.fits.HDUList
        output hdu_list top-level FITS object.
    """

    # get time array
    times = tsdata["TIME"]
    nexps = len(times)
    nsources = np.shape(tsdata["X"])[1]

    # get first and last times
    tstart = Time(times[0], format='jd', scale='utc')
    tstop = Time(times[-1], format='jd', scale='utc')

    # add information about data in the product
    info['ORIGIN'] = ('LNA/MCTI', 'institution responsible for creating this file')
    info['CREATOR'] = ("SPARC4-PIPELINE", 'pipeline job and program used to produc')
    info['FILEVER'] = ('1.0', 'file format version')
    info['DATE'] = (Time.now().iso, 'file creation date')
    info['TSTART'] = (tstart.jd, 'observation start time in BJD')
    info['TSTOP'] = (tstop.jd, 'observation stop time in BJD')
    info['DATE-OBS'] = (tstart.isot, 'TSTART as UTC calendar date')
    info['DATE-END'] = (tstop.isot, 'TSTOP as UTC calendar date')
    info['NEXPS'] = (nexps, 'number of exposures')
    info['NSOURCES'] = (nsources, 'number of sources')

    # create empty header
    header = fits.PrimaryHDU().header

    # add keys given by the info dict
    for key in info.keys() :
        header.set(key, info[key][0], info[key][1])

    # create primary hdu
    primary_hdu = fits.PrimaryHDU(header=header)

    # initialize list of hdus with the primary hdu
    hdu_array = [primary_hdu]

    # initialize big data table
    coords_vars = [tsdata["X"],tsdata["Y"],tsdata["RA"],tsdata["DEC"],tsdata["FWHM"]]
    coords_labs = ["X","Y","RA","DEC","FWHM"]
    # set coords array
    coords_array = packTimeSeriesData(tsdata["TIME"], vars=coords_vars, labs=coords_labs)
    # create time+coords hdu
    coords_hdu = fits.TableHDU(data=coords_array, name='TIME_COORDS')
    # append time+coords hdu
    hdu_array.append(coords_hdu)

    # loop over each key in the tsdata array to create a fits extension for each source
    for key in catalog_names :

        catalog_phot_vars, catalog_phot_labs = [], []

        catalog_data = tsdata[key]

        for varkey in catalog_data.keys() :
            catalog_phot_vars.append(tsdata[key][varkey])
            catalog_phot_labs.append(varkey)

        # set catalog photometry data array
        catalog_phot_array = packTimeSeriesData(tsdata["TIME"], vars=catalog_phot_vars, labs=catalog_phot_labs)

        # create catalog photometry hdu
        catalog_phot_hdu = fits.TableHDU(data=catalog_phot_array, name=key)
        # append hdu
        hdu_array.append(catalog_phot_hdu)

    # create hdu list
    hdu_list = fits.HDUList(hdu_array)

    # write FITS file if filename is given
    if filename != "" :
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list



def readPhotTimeSeriesData(sci_list, catalog_key='CATALOG_PHOT_AP006', longitude=-45.5825, latitude=-22.5344, altitude=1864, time_keyword='DATE-OBS', time_format='isot', time_scale='utc', gettimedata=True, getcoordsdata=True, getphotdata=True) :

    """ Read photometric time series data from a list of images

    Parameters
    ----------
    sci_list : list
        list of file paths for the reduced science image products containing photometric data
    catalog_key : str
        keyword to identify catalog FITS extension
    longitude : float
        East geographic longitude [deg] of observatory; default is OPD longitude of -45.5825 degrees
    latitude : float
        North geographic latitude [deg] of observatory; default is OPD latitude of -22.5344 degrees
    altitude : float
        Observatory elevation [m] above sea level. Default is OPD altitude of 1864 m
    time_keyword : str
        Time keyword in fits header. Default is 'DATE-OBS'
    time_format : str
        Time format in fits header. Default is 'isot'
    time_scale : str
        Time scale in fits header. Default is 'utc'

    Returns
    -------
    tsdata : dict
        output dictionary container for the times series data.
        The following keys are returned in this container:

        tsdata["TIME"] : numpy.ndarray (N)
            float array containing the N times in BJD
        tsdata["X"] : numpy.ndarray (N x M)
            float array containing the N x-coordinates for M sources in the catalog
        tsdata["Y"] : numpy.ndarray (N x M)
            float array containing the N y-coordinates for M sources in the catalog
        tsdata["RA"] : numpy.ndarray (N x M)
            float array containing the N right ascensions for M sources in the catalog
        tsdata["DEC"] : numpy.ndarray (N x M)
            float array containing the N declinations for M sources in the catalog
        tsdata["FWHM"] : numpy.ndarray (N x M)
            float array containing the N full widths at half maximum for M sources in the catalog
        tsdata["CATALOG*"] : dict
            dict container to save photometry data for each catalog extension
        tsdata["CATALOG*"]["MAG"] : numpy.ndarray (N x M)
            float array containing the N magnitudes for M sources in the catalog
        tsdata["CATALOG*"]["EMAG"] : numpy.ndarray (N x M)
            float array containing the N magnitude uncertainties for M sources in the catalog
        tsdata["CATALOG*"]["SKYMAG"] : numpy.ndarray (N x M)
            float array containing the N sky magnitudes for M sources in the catalog
        tsdata["CATALOG*"]["ESKYMAG"] : numpy.ndarray (N x M)
            float array containing the N sky magnitude uncertainties for M sources in the catalog
        tsdata["CATALOG*"]["FLAG"] : numpy.ndarray (N x M)
            uint array containing the N flags for M sources in the catalog
            flag (int):
                0 : single star, all aperture pixels used, no issues in the photometry
                1 : single star, part of pixels in aperture have been rejected, no issues in the photometry
                2 : single star, issues in the photometry
                3 : blended star, all aperture pixels used, no issues in the photometry
                4 : blended star, part of pixels in aperture have been rejected, no issues in the photometry
                5 : blended star, issues in the photometry
                6 : not a star, all aperture pixels used, no issues in the photometry
                7 : not a star, part of pixels in aperture have been rejected, no issues in the photometry
                8 : not a star, issues in the photometry
    """

    altitude = altitude*u.m
    observ_location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=altitude)

    times = []
    xs, ys = [], []
    ras, decs = [], []
    fwhms = []
    mags, emags = [], []
    smags, esmags = [], []
    flags = []

    for i in range(len(sci_list)) :
        #print("image {} of {} -> {}".format(i+1,len(sci_list),sci_list[i]))
        # open sci image product
        hdu_list = fits.open(sci_list[i])
        hdr = deepcopy(hdu_list[0].header)

        try :
            catalog = deepcopy(hdu_list[catalog_key].data)
        except :
            print("WARNING: could not open catalog extension: {} in FITS image: {}, skipping ...".format(catalog_key,sci_list[i]))
            continue

        if gettimedata :
            # set obstime
            obstime=Time(hdr[time_keyword], format=time_format, scale=time_scale, location=observ_location)
            # append JD  to time series
            times.append(obstime.jd)
            # open catalog data

        if getcoordsdata :
            # append all other quantities to time series
            xs.append(catalog['X'])
            ys.append(catalog['Y'])
            ras.append(catalog['RA'])
            decs.append(catalog['DEC'])
            fwhms.append(np.sqrt(catalog['FWHMX']*catalog['FWHMX'] + catalog['FWHMY']*catalog['FWHMY']))

        if getphotdata :
            mags.append(catalog['MAG'])
            emags.append(catalog['EMAG'])
            smags.append(catalog['SKYMAG'])
            esmags.append(catalog['ESKYMAG'])
            flags.append(catalog['FLAG'])

        hdu_list.close()
        del catalog
        del hdu_list

    tsdata = {}

    if gettimedata :

        times = np.array(times, dtype='f8')
        tsdata["TIME"] = times

    if getcoordsdata :

        xs = np.array(xs, dtype='f8')
        ys = np.array(ys, dtype='f8')
        ras = np.array(ras, dtype='f8')
        decs = np.array(decs, dtype='f8')
        fwhms = np.array(fwhms, dtype='f8')

        tsdata["X"] = xs
        tsdata["Y"] = ys
        tsdata["RA"] = ras
        tsdata["DEC"] = decs
        tsdata["FWHM"] = fwhms

    if getphotdata :

        mags = np.array(mags, dtype='f8')
        emags = np.array(emags, dtype='f8')
        smags = np.array(smags, dtype='f8')
        esmags = np.array(esmags, dtype='f8')
        flags = np.array(flags, dtype='i4')

        tsdata["MAG"] = mags
        tsdata["EMAG"] = emags
        tsdata["SKYMAG"] = smags
        tsdata["ESKYMAG"] = esmags
        tsdata["FLAG"] = flags

    return tsdata


def nan_proof_keyword(value) :
    if np.isnan(value) :
        return "NaN"
    elif np.isinf(value) :
        return "inf"
    else :
        return value

def polarProduct(sources, polar_catalogs, info={}, filename="") :
    """ Create a polarimetry FITS product

    Parameters
    ----------
    sources : list
        list of dicts containing the sources catalogs data

    polar_catalogs : list
        list of dicts containing the polarimetry catalogs data

    info : dict
        dictionary with additional header cards to include in the header of product
        The following format must be used:
            info = {key1: (value1, comment1), key2: (value2, comment2), ... }

    filename : str, optional
        The output file name to save product. If empty, file won't be saved.

    Returns
    -------
    hdu_list : astropy.io.fits.HDUList
        output hdu_list top-level FITS object.
    """

    nsources = len(sources)

    #tstart = Time(times[0], format='jd', scale='utc')
    #tstop = Time(times[-1], format='jd', scale='utc')

    # add information about data in the product
    info['ORIGIN'] = ('LNA/MCTI', 'institution responsible for creating this file')
    info['CREATOR'] = ("SPARC4-PIPELINE", 'pipeline job and program used to produc')
    info['FILEVER'] = ('1.0', 'file format version')
    info['DATE'] = (Time.now().iso, 'file creation date')
    info['NSOURCES'] = (nsources, 'number of sources')
    #info['CHANNEL'] = (nsources, 'number of sources')
    #info['BAND'] = (nsources, 'number of sources')

    # create empty header
    header = fits.PrimaryHDU().header

    # add keys given by the info dict
    for key in info.keys() :
        header.set(key, info[key][0], info[key][1])

    # create primary hdu
    primary_hdu = fits.PrimaryHDU(header = header)

    hdus = [primary_hdu]

    for j in range(nsources) :

        # create empty header for catalog extension
        catalog_header = fits.PrimaryHDU().header

        src = sources[j]

        catalog_header.set("SRCINDEX", j, "Source index")
        catalog_header.set("RA", nan_proof_keyword(src['RA']), "right ascension [deg]")
        catalog_header.set("DEC", nan_proof_keyword(src['DEC']), "declination [deg]")

        catalog_header.set("X1", nan_proof_keyword(src['X1']), "x-position of North beam [pix]")
        catalog_header.set("Y1", nan_proof_keyword(src['Y1']), "y-position of North beam [pix]")
        catalog_header.set("X2", nan_proof_keyword(src['X2']), "x-position of South beam [pix]")
        catalog_header.set("Y2", nan_proof_keyword(src['Y2']), "y-position of South beam [pix]")
        catalog_header.set("MAG", nan_proof_keyword(src['MAG']), "magnitude")
        catalog_header.set("EMAG", nan_proof_keyword(src['EMAG']), "magnitude error")
        catalog_header.set("FWHM", nan_proof_keyword(src['FWHM']), "full width at half maximum [pix]")

        # add number of apertures in catalog table
        catalog_header.set("NAPER", len(polar_catalogs), "Number of apertures")

        # collect catalog data
        catdata = []

        for key in polar_catalogs[j].keys() :
            catdata.append(polar_catalogs[j][key])

        # set names and data format for each column in the catalog table
        dtype=[('INDEX', 'i4'), ('APER', 'i4'),
                   ('Q', 'f8'), ('EQ', 'f8'), ('U', 'f8'), ('EU', 'f8'), ('V', 'f8'), ('EV', 'f8'),
                   ('P', 'f8'), ('EP', 'f8'), ('THETA', 'f8'), ('ETHETA', 'f8'),
                   ('K', 'f8'), ('EK', 'f8'), ('ZERO', 'f8'), ('EZERO', 'f8'), ('NOBS', 'i4'), ('NPAR', 'i4'),
                   ('CHI2', 'f8'), ('FLAG', 'i4')]

        for ii in range(header['NEXPS']) :
            dtype.append(("Z{:04d}".format(ii), 'f8'))
            dtype.append(("EZ{:04d}".format(ii), 'f8'))

        # cast catalog data into numpy array
        catalog_array = np.array(catdata, dtype=dtype)

        cat_label = "CATALOG_POLAR_{:05d}".format(j)

        hdu_catalog = fits.TableHDU(data=catalog_array, header=catalog_header, name=cat_label)

        # append catalog hdu to hdulist
        hdus.append(hdu_catalog)

    # create hdu list
    hdu_list = create_hdu_list(hdus)

    # write FITS file if filename is given
    if filename != "" :
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def polarTimeSeriesProduct(tsdata, catalog_names, info={}, filename="") :
    """ Create a polarimetric time series FITS product

    Parameters
    ----------
    tsdata : dict
        time series data container
    catalog_names : list
        list of str with source catalog names
    info : dict
        dictionary with additional header cards to include in the header of product
        The following format must be used:
            info = {key1: (value1, comment1), key2: (value2, comment2), ... }

    filename : str, optional
        The output file name to save product. If empty, file won't be saved.

    Returns
    -------
    hdu_list : astropy.io.fits.HDUList
        output hdu_list top-level FITS object.
    """

    # get time array
    times = tsdata[catalog_names[0]]["TIME"]
    # get number of polar measurements in the time series
    npolmeas = len(times)

    # add information about data in the product
    info['ORIGIN'] = ('LNA/MCTI', 'institution responsible for creating this file')
    info['CREATOR'] = ("SPARC4-PIPELINE", 'pipeline job and program used to produc')
    info['FILEVER'] = ('1.0', 'file format version')
    info['DATE'] = (Time.now().iso, 'file creation date')
    info['NPOLMEAS'] = (npolmeas, 'number of measurements in time series')

    # create empty header
    header = fits.PrimaryHDU().header

    # add keys given by the info dict
    for key in info.keys() :
        header.set(key, info[key][0], info[key][1])

    # create primary hdu
    primary_hdu = fits.PrimaryHDU(header=header)

    # initialize list of hdus with the primary hdu
    hdu_array = [primary_hdu]

    # loop over each key in the tsdata array to create a fits extension for each source
    for key in catalog_names :

        tstbl = Table(tsdata[key])

        # create catalog polarimetry hdu for current source
        catalog_polar_hdu = fits.BinTableHDU(data=tstbl, name=key.replace("CATALOG_POLAR","POLAR_TIMESERIES"))

        # append hdu
        hdu_array.append(catalog_polar_hdu)

    # create hdu list
    hdu_list = fits.HDUList(hdu_array)

    # write FITS file if filename is given
    if filename != "" :
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list

