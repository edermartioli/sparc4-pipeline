"""
    Created on May 2 2022

    Description: Library for the SPARC4 pipeline products

    @author: Eder Martioli <emartioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

import os
from copy import deepcopy
from typing import Collection, Union

import numpy as np
from astropop.framedata import FrameData
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

import sparc4.utils as s4utils

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


def masterCalibration(list_of_imgs, img_data=[], err_data=[], mask_data=[], info={}, filename=""):
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
    for i in range(ninimgs):
        # get file basename and add it to the info dict
        basename = os.path.basename(list_of_imgs[i])
        info['IN{:06d}'.format(i)] = (
            basename, 'input file {} of {}'.format(i, ninimgs))

    # create primary hdu with header of base image
    primary_hdu = fits.PrimaryHDU(header=baseheader)

    # add keys given by the info dict
    for key in info.keys():
        primary_hdu.header.set(key, info[key][0], info[key][1])

    # define default arrays in case they are not provided
    if len(img_data) == 0:
        img_data = np.empty((1024, 1024), dtype=float) * np.nan
    if len(err_data) == 0:
        err_data = np.full_like(img_data, np.nan)
    if len(mask_data) == 0:
        mask_data = np.zeros_like(img_data)

    # set data cube into primary extension
    primary_hdu.data = np.array([img_data, err_data, mask_data])

    # create hdu list
    hdu_list = create_hdu_list([primary_hdu])

    # write FITS file if filename is given
    if filename != "":
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def getFrameFromMasterCalibration(filename):
    """ Pipeline module to get Frame from a master calibration input fits file

    Parameters
    ----------
    input : str
        input fits image path
        
    Returns
    -------
    frame: AStropop Frame
        image data in the Frame format
    """
    hdu_list = fits.open(filename)
    header = hdu_list[0].header

    img_data = hdu_list[0].data[0]
    err_data = hdu_list[0].data[1]
    mask_data = hdu_list[0].data[2]

    unit = None
    if header['BUNIT'] == 'electron':
        unit = u.electron

    frame = FrameData(data=img_data, unit=unit, uncertainty=err_data, mask=mask_data, header=header)

    return frame


def readScienceImageCatalogs(input):
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

    for hdu in hdu_list:
        if hdu.name != 'PRIMARY':
            catdata = hdu.data
            apercat = {}
            for i in range(len(catdata)):
                apercat["{}".format(i)] = catdata[i]
            catalogs.append(apercat)

    return catalogs


def scienceImageProduct(original_image, img_data=[], info={}, catalogs=[], polarimetry=False, filename="", catalog_beam_ids=["S", "N"], wcs_header=None, time_key="DATE-OBS", ra="", dec=""):
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

    if wcs_header:
        baseheader += wcs_header

    baseheader = s4utils.set_timecoords_keys(baseheader, time_key=time_key, ra=ra, dec=dec)

    # create primary hdu with header of base image
    primary_hdu = fits.PrimaryHDU(header=baseheader)

    # add keys given by the info dict
    for key in info.keys():
        primary_hdu.header.set(key, info[key][0], info[key][1])

    # define default arrays in case they are not provided
    if len(img_data) == 0:
        img_data = np.empty((1024, 1024), dtype=float) * np.nan

    # set data cube into primary extension
    primary_hdu.data = np.array(img_data)

    hdus = [primary_hdu]

    for j in range(len(catalogs)):
        # create empty header for catalog extension
        catalog_header = fits.PrimaryHDU().header

        # add number of objects in catalog table
        catalog_header.set("NOBJCAT", len(
            catalogs[j].keys()), "Number of objects in the catalog")

        # collect catalog data
        catdata = []
        for key in catalogs[j].keys():
            catdata.append(catalogs[j][key])

        # set names and data format for each column in the catalog table
        dtype = [('SRCINDEX', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('X', 'f8'), ('Y', 'f8'), ('FWHMX', 'f8'), ('FWHMY','f8'), ('MAG', 'f8'), ('EMAG', 'f8'), ('SKYMAG', 'f8'), ('ESKYMAG', 'f8'), ('APER', 'i4'), ('FLAG', 'i4')]

        # cast catalog data into numpy array
        catalog_array = np.array(catdata, dtype=dtype)

        # get photometry aperture value for the catalog label
        aperture_value = catalog_array[0][11]

        # add aperture value to catalog header
        catalog_header.set("APRADIUS", aperture_value,
                           "Aperture radius in pixels")

        cat_label = "UNLABELED_CATALOG"

        if polarimetry:
            if (j % 2) == 0:
                catalog_header.set("POLBEAM", catalog_beam_ids[0], "Polar beam: [N]orth or [S]outh")
                cat_label = "CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[0], aperture_value)
            else:
                catalog_header.set("POLBEAM", catalog_beam_ids[1], "Polar beam: [N]orth or [S]outh")
                cat_label = "CATALOG_POL_{}_AP{:03d}".format(catalog_beam_ids[1], aperture_value)
        else:
            cat_label = "CATALOG_PHOT_AP{:03d}".format(aperture_value)

        hdu_catalog = fits.TableHDU(data=catalog_array, header=catalog_header, name=cat_label)

        # set each column unit
        column_units = ["", "DEG", "DEG", "PIXEL", "PIXEL","PIXEL", "PIXEL", "MAG", "MAG", "MAG", "MAG", "PIXEL", ""]

        # add description for each column in the header
        for i in range(len(column_units)):
            if column_units[i] != "":
                hdu_catalog.header.comments["TTYPE{:d}".format(i+1)] = "units of {}".format(column_units[i])

        # append catalog hdu to hdulist
        hdus.append(hdu_catalog)

    # create hdu list
    hdu_list = create_hdu_list(hdus)

    # write FITS file if filename is given
    if filename != "":
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def packTimeSeriesData(times, vars=[], labs=[]):
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
    dtype = [('TIME', 'f8')]

    for j in range(nsources):
        for k in range(len(labs)):
            dtype.append(('{}{:08d}'.format(labs[k], j), 'f8'))

    for i in range(len(times)):
        visit = [times[i]]
        for j in range(nsources):
            for k in range(len(labs)):
                visit.append(vars[k][i][j])
        tsarray.append(tuple(visit))

    # cast coordinates data into numpy array
    tsarray = np.array(tsarray, dtype=dtype)

    return tsarray


def photTimeSeriesProduct(tsdata, apertures, info={}, filename=""):
    """ Create a photometric time series FITS product

    Parameters
    ----------
    tsdata : dict
        time series data container
    apertures : dict
        apertures radius in pixels for all existing extensions

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

    # add information about data in the product
    info['ORIGIN'] = (
        'LNA/MCTI', 'institution responsible for creating this file')
    info['CREATOR'] = ("SPARC4-PIPELINE",
                       'pipeline job and program used to produc')
    info['FILEVER'] = ('1.0', 'file format version')
    info['DATE'] = (Time.now().iso, 'file creation date')

    # create empty header
    header = fits.PrimaryHDU().header

    # add keys given by the info dict
    for key in info.keys():
        header.set(key, info[key][0], info[key][1])

    # create primary hdu
    primary_hdu = fits.PrimaryHDU(header=header)

    # initialize list of hdus with the primary hdu
    hdu_array = [primary_hdu]

    # loop over each key in the tsdata array to create a fits extension for each source
    for catalog_key in tsdata.keys():

        # create empty header for catalog extension
        catalog_header = fits.PrimaryHDU().header

        # set aperture radius in header of extension
        catalog_header.set("APRADIUS", apertures[catalog_key], "aperture radius in pixels")

        # create catalog photometry hdu
        catalog_phot_hdu = fits.BinTableHDU(data=tsdata[catalog_key], header=catalog_header, name=catalog_key)

        # append hdu
        hdu_array.append(catalog_phot_hdu)

    # create hdu list
    hdu_list = fits.HDUList(hdu_array)

    # write FITS file if filename is given
    if filename != "":
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def readPhotTimeSeriesData(sci_list,
                           catalog_key='CATALOG_PHOT_AP006',
                           longitude=-45.5825,
                           latitude=-22.5344,
                           altitude=1864,
                           time_keyword='DATE-OBS',
                           time_format='isot',
                           time_scale='utc',
                           time_span_for_rms=5,
                           keys_to_add_header_data=[]) :
                           
    """ Read photometric time series data from a list of images

    Parameters
    ----------
    sci_list : list
        list of file paths for the reduced science image products containing photometric data
    catalog_key : str, optional
        keyword to identify catalog FITS extension
    longitude : float, optional
        East geographic longitude [deg] of observatory; default is OPD longitude of -45.5825 degrees
    latitude : float, optional
        North geographic latitude [deg] of observatory; default is OPD latitude of -22.5344 degrees
    altitude : float, optional
        Observatory elevation [m] above sea level. Default is OPD altitude of 1864 m
    time_keyword : str, optional
        Time keyword in fits header. Default is 'DATE-OBS'
    time_format : str, optional
        Time format in fits header. Default is 'isot'
    time_scale : str, optional
        Time scale in fits header. Default is 'utc'
    time_span_for_rms : float, optional
        Time span (in minutes) around a given observation to include other observations to
        calculate running rms.
    keys_to_add_header_data : list of str, optional
        list of header keywords to get data from and include in time series
    Returns
    -------
    tsdata : astropy.table.Table
        output Table container for the times series data.
        The following keys are returned in this table:

        tsdata["SRCINDEX"] : numpy.ndarray (N)
            source index array
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
        tsdata["MAG"] : numpy.ndarray (N x M)
            float array containing the N magnitudes for M sources in the catalog
        tsdata["EMAG"] : numpy.ndarray (N x M)
            float array containing the N magnitude uncertainties for M sources in the catalog
        tsdata[["SKYMAG"] : numpy.ndarray (N x M)
            float array containing the N sky magnitudes for M sources in the catalog
        tsdata["ESKYMAG"] : numpy.ndarray (N x M)
            float array containing the N sky magnitude uncertainties for M sources in the catalog
        tsdata["FLAG"] : numpy.ndarray (N x M)
            uint array containing the N flags for M sources in the catalog
        tsdata["RMS"] : numpy.ndarray (N x M)
            float array containing the N running rms for M sources in the catalog
        ...
        tsdata["selected header keys"] : numpy.ndarray (N x M)
            float array containing the N running rms for M sources in the catalog
    """

    altitude = altitude*u.m
    observ_location = EarthLocation.from_geodetic(
        lat=latitude, lon=longitude, height=altitude)

    srcindex = np.array([])
    times = np.array([])
    xs, ys = np.array([]), np.array([])
    ras, decs = np.array([]), np.array([])
    fwhms = np.array([])
    mags, emags = np.array([]), np.array([])
    smags, esmags = np.array([]), np.array([])
    flags = np.array([])

    keys_data = []
    for j in range(len(keys_to_add_header_data)) :
        keys_data.append(np.array([]))

    for i in range(len(sci_list)):
        # print("image {} of {} -> {}".format(i+1,len(sci_list),sci_list[i]))
        # open sci image product
        hdu_list = fits.open(sci_list[i])
        hdr = deepcopy(hdu_list[0].header)

        # open catalog data
        try:
            catalog = deepcopy(hdu_list[catalog_key].data)
        except:
            print("WARNING: could not open catalog extension: {} in FITS image: {}, skipping ...".format(
                catalog_key, sci_list[i]))
            continue

        # append source index information
        srcindex = np.append(srcindex, catalog['SRCINDEX'])

        # set obstime
        obstime = Time(hdr[time_keyword], format=time_format,
                       scale=time_scale, location=observ_location)
        # append JD  to time series
        times = np.append(times, np.full_like(catalog['SRCINDEX'], obstime.jd, dtype=float))

        # append coordinates information
        ras = np.append(ras, catalog['RA'])
        decs = np.append(decs, catalog['DEC'])
        xs = np.append(xs, catalog['X'])
        ys = np.append(ys, catalog['Y'])

        # append fwhms
        fwhms = np.append(fwhms, np.sqrt(
            catalog['FWHMX']*catalog['FWHMX'] + catalog['FWHMY']*catalog['FWHMY']))

        # append photometric information
        mags = np.append(mags, catalog['MAG'])
        emags = np.append(emags, catalog['EMAG'])
        smags = np.append(smags, catalog['SKYMAG'])
        esmags = np.append(esmags, catalog['ESKYMAG'])
        flags = np.append(flags, catalog['FLAG'])

        for j in range(len(keys_to_add_header_data)) :
            keys_data[j] = np.append(keys_data[j], np.full_like(catalog['SRCINDEX'], float(hdr[keys_to_add_header_data[j]]), dtype=float))

        hdu_list.close()
        del catalog
        del hdu_list

    tsdata = {}

    tsdata["TIME"] = times
    tsdata["SRCINDEX"] = srcindex
    tsdata["RA"] = ras
    tsdata["DEC"] = decs
    tsdata["X"] = xs
    tsdata["Y"] = ys
    tsdata["FWHM"] = fwhms
    tsdata["MAG"] = mags
    tsdata["EMAG"] = emags
    tsdata["SKYMAG"] = smags
    tsdata["ESKYMAG"] = esmags
    tsdata["FLAG"] = flags
    tsdata["RMS"] = np.full_like(tsdata["TIME"], np.nan)
        
    # Below we calculate a running rms for each source's data
    # get number of targets from header of first image
    nsrc = fits.getheader(sci_list[0], 1)['NOBJCAT']

    # convert time span from minutes to days
    time_span_for_rms_d = time_span_for_rms/(60*24)

    # populate rms data
    for i in range(nsrc):
        keep = srcindex == i
        t = times[keep]
        mag, emag = mags[keep], emags[keep]

        for j in range(len(t)):

            weights = 1/(emag*emag)

            keep_for_rms = (t > t[j] - time_span_for_rms_d /
                            2) & (t < t[j] + time_span_for_rms_d/2)

            keep_for_rms &= np.isfinite(mag)
            keep_for_rms &= np.isfinite(weights)

            if len(mag[keep_for_rms]) > 1:
                rms = np.sqrt(
                    np.cov(mag[keep_for_rms], aweights=weights[keep_for_rms]))
                tsdata["RMS"][i+nsrc*j] = rms
            elif len(mag[keep_for_rms]) == 1:
                tsdata["RMS"][i+nsrc*j] = tsdata["EMAG"][i+nsrc*j]

    for j in range(len(keys_to_add_header_data)) :
        tsdata[keys_to_add_header_data[j]] = keys_data[j]

    return Table(tsdata)


def polarProduct(polar_catalogs, info={}, filename=""):
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

    # add information about data in the product
    info['ORIGIN'] = (
        'LNA/MCTI', 'institution responsible for creating this file')
    info['CREATOR'] = ("SPARC4-PIPELINE",
                       'pipeline job and program used to produc')
    info['FILEVER'] = ('1.0', 'file format version')
    info['DATE'] = (Time.now().iso, 'file creation date')

    # create empty header
    header = fits.PrimaryHDU().header

    # add keys given by the info dict
    for key in info.keys():
        header.set(key, info[key][0], info[key][1])

    # create primary hdu
    primary_hdu = fits.PrimaryHDU(header=header)

    # add primary hdu into a list of hdus
    hdus = [primary_hdu]

    for aperture_key in polar_catalogs.keys():

        # create empty header for catalog extension
        catalog_header = fits.PrimaryHDU().header

        # set aperture radius in header of extension
        catalog_header.set("APRADIUS", polar_catalogs[aperture_key]["APER"][0], "aperture radius in pixels")

        # cast polar catalog dictionary into an astropy Table
        tbl_catalog = Table(polar_catalogs[aperture_key])

        # stare in a catalog hdu
        hdu_catalog = fits.BinTableHDU(data=tbl_catalog, header=catalog_header, name=aperture_key)

        # append catalog hdu to hdulist
        hdus.append(hdu_catalog)

    # create hdu list
    hdu_list = create_hdu_list(hdus)

    # write FITS file if filename is given
    if filename != "":
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def polarTimeSeriesProduct(tsdata, info={}, filename=""):
    """ Create a polarimetric time series FITS product

    Parameters
    ----------
    tsdata : dict
        time series data container
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
    times = tsdata["TIME"][tsdata["SRCINDEX"] == 0]
    # get number of polar measurements in the time series
    npolmeas = len(times)

    # add information about data in the product
    info['ORIGIN'] = (
        'LNA/MCTI', 'institution responsible for creating this file')
    info['CREATOR'] = ("SPARC4-PIPELINE",
                       'pipeline job and program used to produc')
    info['FILEVER'] = ('1.0', 'file format version')
    info['DATE'] = (Time.now().iso, 'file creation date')
    info['NPOLMEAS'] = (npolmeas, 'number of measurements in time series')

    # create empty header
    header = fits.PrimaryHDU().header

    # add keys given by the info dict
    for key in info.keys():
        header.set(key, info[key][0], info[key][1])

    # create primary hdu
    primary_hdu = fits.PrimaryHDU(header=header)

    # initialize list of hdus with the primary hdu
    hdu_array = [primary_hdu]

    # create catalog polarimetry hdu for current source
    catalog_polar_hdu = fits.BinTableHDU(
        data=Table(tsdata), name="POLAR_TIMESERIES")

    # append hdu
    hdu_array.append(catalog_polar_hdu)

    # create hdu list
    hdu_list = fits.HDUList(hdu_array)

    # write FITS file if filename is given
    if filename != "":
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list
