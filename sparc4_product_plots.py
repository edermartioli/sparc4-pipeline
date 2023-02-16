# -*- coding: iso-8859-1 -*-
"""
    Created on Jun 7 2022

    Description: Library for plotting SPARC4 pipeline products
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

import warnings
from copy import deepcopy

from astropop.math.physical import QFloat

def plot_cal_frame(filename, output="", percentile=99.5, xcut=512, ycut=512) :

    """ Plot calibration (bias, flat) frame
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    hdul = fits.open(filename)

    img_data = hdul["PRIMARY"].data[0]
    err_data = hdul["PRIMARY"].data[1]
    #mask_data = hdul["PRIMARY"].data[2]

    img_mean = QFloat(np.mean(img_data),np.std(img_data))
    noise_mean = QFloat(np.mean(err_data),np.std(err_data))

    # plot best polarimetry results
    fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex=False, sharey=False, gridspec_kw={'hspace': 0.5, 'height_ratios': [4, 1, 1]})

    axes[0,0].set_title("image: mean:{}".format(img_mean))
    axes[0,0].imshow(img_data, vmin=np.percentile(img_data,100 - percentile), vmax=np.percentile(img_data, percentile), origin='lower')
    axes[0,0].set_xlabel("columns (pixel)", fontsize=16)
    axes[0,0].set_ylabel("rows (pixel)", fontsize=16)
    
    xsize, ysize = np.shape(img_data)
    x, y = np.arange(xsize), np.arange(ysize)
    
    axes[1,0].plot(x, img_data[ycut,:])
    axes[1,0].set_ylabel("flux".format(ycut), fontsize=16)
    axes[1,0].set_xlabel("columns (pixel)", fontsize=16)
    
    axes[2,0].plot(y, img_data[:,xcut])
    axes[2,0].set_ylabel("flux".format(xcut), fontsize=16)
    axes[2,0].set_xlabel("rows (pixel)", fontsize=16)

    axes[0,1].set_title("noise: mean:{}".format(noise_mean))
    axes[0,1].imshow(err_data, vmin=np.percentile(err_data,100 - percentile), vmax=np.percentile(err_data, percentile), origin='lower')
    axes[0,1].set_xlabel("columns (pixel)", fontsize=16)
    axes[0,1].set_ylabel("rows (pixel)", fontsize=16)

    axes[1,1].plot(x, err_data[ycut,:])
    axes[1,1].set_ylabel(r"$\sigma$".format(ycut), fontsize=16)
    axes[1,1].set_xlabel("columns (pixel)", fontsize=16)
    
    axes[2,1].plot(y, err_data[:,xcut])
    axes[2,1].set_ylabel(r"$\sigma$".format(xcut), fontsize=16)
    axes[2,1].set_xlabel("rows (pixel)", fontsize=16)

    plt.show()



def plot_sci_frame(filename, cat_ext=9, nstars=5, output="", percentile=98, use_sky_coords=False) :

    """ Plot science frame
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    hdul = fits.open(filename)
    img_data = hdul["PRIMARY"].data[0] 
    #err_data = hdul["PRIMARY"].data[1]
    #mask_data = hdul["PRIMARY"].data[2]


    x, y = hdul[cat_ext].data['x'], hdul[cat_ext].data['y']
    mean_aper = np.mean(hdul[cat_ext].data['APER'])

    if nstars > len(x) :
        nstars = len(x)
        
    fig = plt.figure(figsize=(10, 10))

    if use_sky_coords :
        # load WCS from image header
        wcs_obj = WCS(hdul[0].header,naxis=2)
        
        # calculate pixel scale
        pixel_scale = proj_plane_pixel_scales(wcs_obj)

        # assume  the N-S component of the pixel scale
        pixel_scale = (pixel_scale[1] * 3600)
        
        ax = plt.subplot(projection=wcs_obj)
        ax.imshow(img_data, vmin=np.percentile(img_data, 100. - percentile), vmax=np.percentile(img_data, percentile), origin='lower', cmap='cividis', aspect='equal')

        world = wcs_obj.pixel_to_world(x, y)
        wx, wy = world.ra.degree, world.dec.degree

        for i in range(len(wx)) :
            sky_center = SkyCoord(wx[i], wy[i], unit='deg')
            sky_radius = Angle(mean_aper*pixel_scale, 'arcsec')
            sky_region = CircleSkyRegion(sky_center, sky_radius)
            pixel_region = sky_region.to_pixel(wcs_obj)
            pixel_region.plot(ax=ax, color='white', lw=2.0)
            if i < nstars :
                text_offset = 0.004
                plt.text(wx[i]+text_offset, wy[i], '{}'.format(i), c='darkred', weight='bold', fontsize=18, transform=ax.get_transform('icrs'))

        plt.xlabel(r'RA')
        plt.ylabel(r'Dec')
        overlay = ax.get_coords_overlay('icrs')
        overlay.grid(color='white', ls='dotted')
    
    else :
    
        plt.plot(x, y, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
        plt.plot(x[:nstars], y[:nstars], 'k+', ms=2*mean_aper/3, lw=1.0, alpha=0.7)
        for i in range(nstars) :
            plt.text(x[i]+1.1*mean_aper, y[i], '{}'.format(i), c='darkred', weight='bold', fontsize=18)
        plt.imshow(img_data, vmin=np.percentile(img_data, 100. - percentile), vmax=np.percentile(img_data, percentile), origin='lower')
        plt.xlabel("columns (pixel)", fontsize=16)
        plt.ylabel("rows (pixel)", fontsize=16)
    
    
    plt.show()



def plot_sci_polar_frame(filename, percentile=99.5) :
    """ Plot science polar frame
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """
    hdul = fits.open(filename)
    img_data = hdul["PRIMARY"].data[0]
    #err_data = hdul["PRIMARY"].data[1]
    #mask_data = hdul["PRIMARY"].data[2]

    x_o, y_o = hdul["CATALOG_POL_N_AP010"].data['x'], hdul["CATALOG_POL_N_AP010"].data['y']
    x_e, y_e = hdul["CATALOG_POL_S_AP010"].data['x'], hdul["CATALOG_POL_S_AP010"].data['y']
    
    mean_aper = np.mean(hdul["CATALOG_POL_N_AP010"].data['APER'])

    plt.figure(figsize=(10, 10))

    plt.plot(x_o, y_o, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
    plt.plot(x_e, y_e, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)

    plt.imshow(img_data, vmin=np.percentile(img_data, 100-percentile), vmax=np.percentile(img_data, percentile), origin='lower')
    
    for i in range(len(x_o)):
        x = [x_o[i], x_e[i]]
        y = [y_o[i], y_e[i]]
        plt.plot(x, y, 'w-o', alpha=0.5)
        plt.annotate(f"{i}", [np.mean(x)-25, np.mean(y)+25], color='w')
        
    plt.xlabel("columns (pixel)", fontsize=16)
    plt.ylabel("rows (pixel)", fontsize=16)
    
    plt.show()
    

def plot_light_curve(filename, target=0, comps=[], output="", nsig=100, plot_sum=True, plot_comps=False) :
        
    """ Plot light curve
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    hdul = fits.open(filename)

    time = hdul["DIFFPHOTOMETRY"].data['TIME']
    
    mintime, maxtime = np.min(time), np.max(time)
    
    data = hdul["DIFFPHOTOMETRY"].data
    
    nstars = int((len(data.columns) - 1) / 2)
    
    warnings.filterwarnings('ignore')
    
    offset = 0.
    for i in range(nstars) :
        lc = -data['DMAG{:06d}'.format(i)]
        elc = data['EDMAG{:06d}'.format(i)]
        
        mlc = np.nanmedian(lc)
        rms = np.nanmedian(np.abs(lc-mlc)) / 0.67449
        #rms = np.nanstd(lc-mlc)
        
        keep = elc < nsig*rms
                
        if i == 0 :
            if plot_sum :
                comp_label = "SUM"
                plt.errorbar(time[keep], lc[keep], yerr=elc[keep], fmt='k.', alpha=0.8, label=r"{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comp_label, mlc, rms*1000))
                plt.plot(time[keep], lc[keep], "k-", lw=0.5)
            
            offset = np.nanpercentile(lc[keep], 1.0) - 4.0*rms
            
            plt.hlines(offset, mintime, maxtime, colors='k', linestyle='-', lw=0.5)
            plt.hlines(offset-rms, mintime, maxtime, colors='k', linestyle='--', lw=0.5)
            plt.hlines(offset+rms, mintime, maxtime, colors='k', linestyle='--', lw=0.5)
        
        else :
            if plot_comps :
                comp_label = "C{:03d}".format(comps[i-1])
                plt.errorbar(time[keep], (lc[keep]-mlc)+offset, yerr=elc[keep], fmt='.', alpha=0.5, label=r"{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comp_label, mlc, rms*1000))
            
        #print(i, mlc, rms)

    plt.xlabel(r"time (BJD)", fontsize=16)
    plt.ylabel(r"$\Delta$mag", fontsize=16)
    plt.legend(fontsize=10)
    plt.show()
