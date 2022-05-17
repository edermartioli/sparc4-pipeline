# -*- coding: iso-8859-1 -*-
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
from astropy.time import Time

from uncertainties import ufloat

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



def scienceImageProduct(original_image, img_data=[], err_data=[], mask_data=[], info={}, catalog={}, filename="") :
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
    catalog : dict
        dictionary with star catalog
        The following format must be used:
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

    filename : str, optional
        The output file name to save product. If empty, file won't be saved.

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

    # create empty header for catalog extension
    catalog_header = fits.PrimaryHDU().header

    # add number of objects in catalog table
    catalog_header.set("NOBJCAT", len(catalog.keys()), "Number of objects in the catalog")

    # collect catalog data
    catdata = []
    for key in catalog.keys() :
        catdata.append(catalog[key])
    
    # set names and data format for each column in the catalog table
    dtype=[('INDEX', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('X', 'f8'), ('Y', 'f8'), ('FWHMX', 'f8'), ('FWHMY', 'f8'), ('MAG', 'f8'), ('EMAG', 'f8'), ('SKYMAG', 'f8'), ('ESKYMAG', 'f8'), ('APER', 'i4'), ('FLAG', 'i4')]
    
    # cast catalog data into numpy array
    catalog_array = np.array(catdata, dtype=dtype)
              
    # create catalog hdu
    hdu_catalog = fits.TableHDU(data=catalog_array, header=catalog_header, name='CATALOG')
    
    column_units = ["", "DEG", "DEG", "PIXEL", "PIXEL", "PIXEL", "PIXEL", "MAG", "MAG", "MAG", "MAG", "PIXEL", ""]
    # add description for each column in the header

    for i in range(len(column_units)) :
        if column_units[i] != "" :
            hdu_catalog.header.comments["TTYPE{:d}".format(i+1)] = "units of {}".format(column_units[i])
    
    # create hdu list
    hdu_list = create_hdu_list([primary_hdu,hdu_catalog])
    
    # write FITS file if filename is given
    if filename != "" :
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list
    

def packTimeSeriesData(times, x, y=[], xlabel="X", ylabel="Y") :
    """ Pack time series data into a numpy array ready to be saved into a FITS Table HDU
    
    Parameters
    ----------
    times : numpy.ndarray (N)
        float array containing the N times in BJD
    x : numpy.ndarray (N x M)
        float array containing the N points in the time series for M sources
    y : numpy.ndarray (N x M) - optional
        float array containing the N points in the time series for M sources
    xlabel : str
        string for the ID name of variable x
    ylabel : str
        string for the ID name of variable y

    Returns
    -------
    tsarray : numpy.recarray (N x (2 x M + 1))
        output tsarray time series array
    """
    
    # get number of sources
    nsources = np.shape(x)[1]
        
    # set array with times
    tsarray = []

    hasyvariable = False
    if len(y) :
        hasyvariable = True

    # set names and data format for each column in the catalog table
    dtype=[('TIME', 'f8')]
    
    for j in range(nsources) :
        dtype.append(('{}{:06d}'.format(xlabel,j), 'f8'))
        if hasyvariable :
            dtype.append(('{}{:06d}'.format(ylabel,j), 'f8'))

    for i in range(len(times)) :
        visit = [times[i]]
        for j in range(nsources) :
            visit.append(x[i][j])
            if hasyvariable :
                visit.append(y[i][j])
        tsarray.append(tuple(visit))
    
    # cast coordinates data into numpy array
    tsarray = np.array(tsarray, dtype=dtype)
    
    return tsarray


def photTimeSeriesProduct(times, ras, decs, mags, emags, smags, esmags, dmags, edmags, flags, info={}, filename="") :
    """ Create a photometric time series FITS product
    
    Parameters
    ----------
    times : numpy.ndarray (N)
        float array containing the N times in BJD
    ras : numpy.ndarray (N x M)
        float array containing the N right ascensions for M sources in the catalog
    decs : numpy.ndarray (N x M)
        float array containing the N declinations for M sources in the catalog
    mags : numpy.ndarray (N x M)
        float array containing the N magnitudes for M sources in the catalog
    emags : numpy.ndarray (N x M)
        float array containing the N magnitude uncertainties for M sources in the catalog
    smags : numpy.ndarray (N x M)
        float array containing the N sky magnitudes for M sources in the catalog
    esmags : numpy.ndarray (N x M)
        float array containing the N sky magnitude uncertainties for M sources in the catalog
    dmags : numpy.ndarray (N x M)
        float array containing the N differential magnitudes for M sources in the catalog
    edmags : numpy.ndarray (N x M)
        float array containing the N differential magnitude uncertainties for M sources in the catalog
    flags : numpy.ndarray (N x M)
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

    nexps = len(times)
    nsources = np.shape(mags)[1]

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
    info['EXT1'] = ("COORDINATES", 'Time [BJD], RA [DEG], Dec [DEG]')
    info['EXT2'] = ("PHOTOMETRY", 'Time [BJD], Magnitude [MAG], Error [MAG]')
    info['EXT3'] = ("SKYPHOTOMETRY", 'Time [BJD], SkyMagnitude [MAG], Error [MAG]')
    info['EXT4'] = ("DIFFPHOTOMETRY", 'Time [BJD], DiffMag [MAG], Error [MAG]')
    info['EXT5'] = ("FLAGS", 'Time [BJD], Flags')

    # create empty header
    header = fits.PrimaryHDU().header

    # add keys given by the info dict
    for key in info.keys() :
        header.set(key, info[key][0], info[key][1])

    # create primary hdu
    primary_hdu = fits.PrimaryHDU(header = header)
    
    # set array of coordinates
    coords_array = packTimeSeriesData(times, ras, decs, xlabel="RA", ylabel="DEC")
    
    # create coordinates hdu
    hdu_coords = fits.TableHDU(data=coords_array, name='COORDINATES')
    
    # set array of magnitudes
    magnitudes_array = packTimeSeriesData(times, mags, emags, xlabel="MAG", ylabel="EMAG")
    # create raw photometry hdu
    hdu_magnitudes = fits.TableHDU(data=magnitudes_array, name='PHOTOMETRY')
    
    # set array of sky magnitudes
    skymags_array = packTimeSeriesData(times, smags, esmags, xlabel="SKYMAG", ylabel="ESKYMAG")
    # create sky photometry hdu
    hdu_skymags = fits.TableHDU(data=skymags_array, name='SKYPHOTOMETRY')
    
    # set array of differential magnitudes
    diffmags_array = packTimeSeriesData(times, dmags, edmags, xlabel="DMAG", ylabel="EDMAG")
    # create differential photometry hdu
    hdu_diffmags = fits.TableHDU(data=diffmags_array, name='DIFFPHOTOMETRY')
    
    # set array of flags
    flags_array = packTimeSeriesData(times, flags, xlabel="FLAG")
    # create flags hdu
    hdu_flags = fits.TableHDU(data=flags_array, name='FLAGS')
    
    # create hdu list
    hdu_list = create_hdu_list([primary_hdu,hdu_coords,hdu_magnitudes,hdu_skymags,hdu_diffmags,hdu_flags])
    
    # write FITS file if filename is given
    if filename != "" :
        hdu_list.writeto(filename, overwrite=True, output_verify="fix+warn")

    # return hdu list
    return hdu_list


def readPhotTimeSeriesData(sci_list) :
    """ Read photometric time series data from a list of images
    
    Parameters
    ----------
    sci_list : list
        list of file paths for the reduced science image products containing photometric data
        
    Returns
    -------
    tsdata : dict
        output dictionary container for the times series data.
        The following keys are returned in this container:

        tsdata["TIME"] : numpy.ndarray (N)
            float array containing the N times in BJD
        tsdata["RA"] : numpy.ndarray (N x M)
            float array containing the N right ascensions for M sources in the catalog
        tsdata["DEC"] : numpy.ndarray (N x M)
            float array containing the N declinations for M sources in the catalog
        tsdata["MAG"] : numpy.ndarray (N x M)
            float array containing the N magnitudes for M sources in the catalog
        tsdata["EMAG"] : numpy.ndarray (N x M)
            float array containing the N magnitude uncertainties for M sources in the catalog
        tsdata["SKYMAG"] : numpy.ndarray (N x M)
            float array containing the N sky magnitudes for M sources in the catalog
        tsdata["ESKYMAG"] : numpy.ndarray (N x M)
            float array containing the N sky magnitude uncertainties for M sources in the catalog
        tsdata["DMAG"] : numpy.ndarray (N x M)
            float array containing the N differential magnitudes for M sources in the catalog
        tsdata["EDMAG"] : numpy.ndarray (N x M)
            float array containing the N differential magnitude uncertainties for M sources in the catalog
        tsdata["FLAG"] : numpy.ndarray (N x M)
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

    times = []
    ras, decs = [], []
    mags, emags = [], []
    smags, esmags = [], []
    dmags, edmags = [], []
    flags = []

    for i in range(len(sci_list)) :
    
        hdu_list = fits.open(sci_list[i])
        
        bjd = hdu_list[0].header["JD"]
        
        times.append(bjd)

        catalog = hdu_list['CATALOG'].data
        
        idmags, iedmags = [], []
        for j in range(len(catalog['MAG'])) :
            usums = ufloat(0,0)
            for k in range(len(catalog['MAG'])) :
                if k != j :
                    usums += ufloat(catalog['MAG'][k],catalog['EMAG'][k])
            udmag = ufloat(catalog['MAG'][j],catalog['EMAG'][j]) / usums
            idmags.append(udmag.nominal_value)
            iedmags.append(udmag.std_dev)
        idmags = np.array(idmags, dtype='f8')
        iedmags = np.array(iedmags, dtype='f8')

        ras.append(catalog['RA'])
        decs.append(catalog['DEC'])
        mags.append(catalog['MAG'])
        emags.append(catalog['EMAG'])
        smags.append(catalog['SKYMAG'])
        esmags.append(catalog['ESKYMAG'])
        dmags.append(idmags)
        edmags.append(iedmags)
        flags.append(catalog['FLAG'])
    
    times = np.array(times, dtype='f8')
    ras = np.array(ras, dtype='f8')
    decs = np.array(decs, dtype='f8')
    mags = np.array(mags, dtype='f8')
    emags = np.array(emags, dtype='f8')
    smags = np.array(smags, dtype='f8')
    esmags = np.array(esmags, dtype='f8')
    dmags = np.array(dmags, dtype='f8')
    edmags = np.array(edmags, dtype='f8')
    flags = np.array(flags, dtype='i4')

    tsdata = {}
    tsdata["TIME"] = times
    tsdata["RA"] = ras
    tsdata["DEC"] = decs
    tsdata["MAG"] = mags
    tsdata["EMAG"] = emags
    tsdata["SKYMAG"] = smags
    tsdata["ESKYMAG"] = esmags
    tsdata["DMAG"] = dmags
    tsdata["EDMAG"] = edmags
    tsdata["FLAG"] = flags

    return tsdata


def polarTimeSeriesProduct() :
    pass
    
def calibS4ImageProduct() :
    pass

def scienceS4ImageProduct() :
    pass

def photS4TimeSeriesProduct() :
    pass
    
def polarS4TimeSeriesProduct() :
    pass
