"""
    Created on Jun 7 2022

    Description: Library for plotting SPARC4 pipeline products

    @author: Eder Martioli <emartioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropop.math.physical import QFloat
from astropop.polarimetry import halfwave_model, quarterwave_model
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from regions import CircleSkyRegion

import sparc4.pipeline_lib as s4pipelib

def plot_cal_frame(filename, percentile=99.5, xcut=512, ycut=512,
                   combine_rows=False, combine_cols=False,
                   combine_method="mean", output="",figsize=(16, 8)):
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
    # mask_data = hdul["PRIMARY"].data[2]

    img_mean = QFloat(np.mean(img_data), np.std(img_data))
    noise_mean = QFloat(np.mean(err_data), np.std(err_data))

    # plot best polarimetry results
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=False, sharey=False, gridspec_kw={
                             'hspace': 0.5, 'height_ratios': [4, 1, 1]})

    axes[0, 0].set_title("image: mean:{}".format(img_mean))
    axes[0, 0].imshow(img_data, vmin=np.percentile(img_data, 100 - percentile),
                      vmax=np.percentile(img_data, percentile), origin='lower')
    axes[0, 0].set_xlabel("columns (pixel)", fontsize=16)
    axes[0, 0].set_ylabel("rows (pixel)", fontsize=16)

    xsize, ysize = np.shape(img_data)
    x, y = np.arange(xsize), np.arange(ysize)

    if combine_rows:
        crow = None
        if combine_method == "mean":
            crow = np.nanmean(img_data, axis=0)
        elif combine_method == "median":
            crow = np.nanmedian(img_data, axis=0)
        else:
            print("ERROR: combine method must be mean or median, exiting ...")
            exit()
        axes[1, 0].plot(x, crow)
        axes[1, 0].set_ylabel("{} flux".format(combine_method), fontsize=16)
    else:
        axes[1, 0].plot(x, img_data[ycut, :])
        axes[1, 0].set_ylabel("flux".format(ycut), fontsize=16)
    axes[1, 0].set_xlabel("columns (pixel)", fontsize=16)

    if combine_cols:
        ccol = None
        if combine_method == "mean":
            ccol = np.nanmean(img_data, axis=1)
        elif combine_method == "median":
            ccol = np.nanmedian(img_data, axis=1)
        else:
            print("ERROR: combine method must be mean or median, exiting ...")
            exit()
        axes[2, 0].plot(y, ccol)
        axes[2, 0].set_ylabel("{} flux".format(combine_method), fontsize=16)
    else:
        axes[2, 0].plot(y, img_data[:, xcut])
        axes[2, 0].set_ylabel("flux".format(xcut), fontsize=16)
    axes[2, 0].set_xlabel("rows (pixel)", fontsize=16)

    axes[0, 1].set_title("noise: mean:{}".format(noise_mean))
    axes[0, 1].imshow(err_data, vmin=np.percentile(err_data, 100 - percentile),
                      vmax=np.percentile(err_data, percentile),
                      origin='lower')
    axes[0, 1].set_xlabel("columns (pixel)", fontsize=16)
    axes[0, 1].set_ylabel("rows (pixel)", fontsize=16)

    axes[1, 1].plot(x, err_data[ycut, :])
    axes[1, 1].set_ylabel(r"$\sigma$".format(ycut), fontsize=16)
    axes[1, 1].set_xlabel("columns (pixel)", fontsize=16)

    axes[2, 1].plot(y, err_data[:, xcut])
    axes[2, 1].set_ylabel(r"$\sigma$".format(xcut), fontsize=16)
    axes[2, 1].set_xlabel("rows (pixel)", fontsize=16)

    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()
        

def plot_sci_frame(filename, cat_ext=3, nstars=5, percentile=98, use_sky_coords=False, output="", figsize=(10, 10), toplabel="", bottomlabel=""):
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
    img_data = hdul["PRIMARY"].data
    # img_data = hdul["PRIMARY"].data[0]
    # err_data = hdul["PRIMARY"].data[1]
    # mask_data = hdul["PRIMARY"].data[2]

    x, y = hdul[cat_ext].data['x'], hdul[cat_ext].data['y']
    mean_aper = np.mean(hdul[cat_ext].data['APER'])

    if nstars > len(x):
        nstars = len(x)

    fig = plt.figure(figsize=figsize)
        
    if use_sky_coords:
        # load WCS from image header
        wcs_obj = WCS(hdul[0].header, naxis=2)

        # calculate pixel scale
        pixel_scale = proj_plane_pixel_scales(wcs_obj)

        # assume the N-S component of the pixel scale and convert to arcsec/pixel
        pixel_scale = (pixel_scale[1] * 3600)
        #print("Pixel scale in Dec: {} arcsec/pixel".format(pixel_scale))

        ax = plt.subplot(projection=wcs_obj)
        ax.imshow(img_data, vmin=np.percentile(img_data, 100. - percentile),
                  vmax=np.percentile(img_data, percentile), origin='lower',
                  cmap='cividis', aspect='equal')

        world = wcs_obj.pixel_to_world(x, y)
        wx, wy = world.ra.degree, world.dec.degree

        for i in range(len(wx)):
            sky_center = SkyCoord(wx[i], wy[i], unit='deg')
            sky_radius = Angle(mean_aper*pixel_scale, 'arcsec')
            sky_region = CircleSkyRegion(sky_center, sky_radius)
            pixel_region = sky_region.to_pixel(wcs_obj)
            pixel_region.plot(ax=ax, color='white', lw=2.0)
            if i < nstars:
                text_offset = 0.004
                plt.text(wx[i]+text_offset, wy[i], '{}'.format(i), c='darkred',
                         weight='bold', fontsize=18,
                         transform=ax.get_transform('icrs'))

        plt.xlabel(r'rA', fontsize=14)
        plt.ylabel(r'dec', fontsize=14)
        overlay = ax.get_coords_overlay('icrs')
        overlay.grid(color='white', ls='dotted')

    else:

        plt.plot(x, y, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
        plt.plot(x[:nstars], y[:nstars], 'k+',ms=2*mean_aper/3, lw=1.0, alpha=0.7)
        for i in range(nstars):
            plt.text(x[i]+1.1*mean_aper, y[i], '{}'.format(i),c='darkred', weight='bold', fontsize=18)
        plt.imshow(img_data, vmin=np.percentile(img_data, 100. - percentile),vmax=np.percentile(img_data, percentile), origin='lower')
        plt.xlabel("columns (pixel)", fontsize=18)
        plt.ylabel("rows (pixel)", fontsize=18)

    if toplabel != "" :
        xmax, ymax = np.shape(img_data)
        plt.annotate(toplabel, [xmax-500, ymax-40], color='w', fontsize=17, alpha=0.75)
        
    if bottomlabel != "" :
        xmax, ymax = np.shape(img_data)
        plt.annotate(bottomlabel, [xmax-550, 10], color='w', fontsize=14, alpha=0.75)

    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()


def plot_sci_polar_frame(filename, ord_catalog_name="CATALOG_POL_N_AP010", ext_catalog_name="CATALOG_POL_S_AP010", percentile=99.5, output="", figsize=(10, 10), toplabel="", bottomlabel=""):
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
    img_data = hdul["PRIMARY"].data
    # img_data = hdul["PRIMARY"].data[0]
    # err_data = hdul["PRIMARY"].data[1]
    # mask_data = hdul["PRIMARY"].data[2]

    x_o, y_o = hdul[ord_catalog_name].data['x'], hdul[ord_catalog_name].data['y']
    x_e, y_e = hdul[ext_catalog_name].data['x'], hdul[ext_catalog_name].data['y']

    mean_aper = np.mean(hdul[ord_catalog_name].data['APER'])

    fig = plt.figure(figsize=figsize)

    plt.plot(x_o, y_o, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
    plt.plot(x_e, y_e, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)

    plt.imshow(img_data, vmin=np.percentile(img_data, 100-percentile),
               vmax=np.percentile(img_data, percentile), origin='lower')

    for i in range(len(x_o)):
        x = [x_o[i], x_e[i]]
        y = [y_o[i], y_e[i]]
        plt.plot(x, y, 'w-o', alpha=0.5)
        plt.annotate(f"{i}", [np.mean(x)-25, np.mean(y)+25], color='w')

    if toplabel != "" :
        xmax, ymax = np.shape(img_data)
        plt.annotate(toplabel, [xmax-500, ymax-40], color='w', fontsize=17, alpha=0.75)
        
    if bottomlabel != "" :
        xmax, ymax = np.shape(img_data)
        plt.annotate(bottomlabel, [xmax-550, 10], color='w', fontsize=14, alpha=0.75)

    plt.xlabel("columns (pixel)", fontsize=16)
    plt.ylabel("rows (pixel)", fontsize=16)

    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()
        

def plot_diff_light_curve(filename, target=0, comps=[], nsig=100,
                          plot_sum=True, plot_comps=False, output="",figsize=(10, 10)):
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

    fig = plt.figure(figsize=figsize)

    hdul = fits.open(filename)
    # Below is a hack to avoid very high values of time in some bad data

    time = hdul["DIFFPHOTOMETRY"].data['TIME']

    mintime, maxtime = np.min(time), np.max(time)

    data = hdul["DIFFPHOTOMETRY"].data

    nstars = int((len(data.columns) - 1) / 2)

    warnings.filterwarnings('ignore')

    offset = 0.
    for i in range(nstars):
        lc = -data['DMAG{:06d}'.format(i)][keeplowtimes]
        elc = data['EDMAG{:06d}'.format(i)][keeplowtimes]

        mlc = np.nanmedian(lc)
        rms = np.nanmedian(np.abs(lc-mlc)) / 0.67449
        # rms = np.nanstd(lc-mlc)

        keep = elc < nsig*rms

        if i == 0:
            if plot_sum:
                comp_label = "SUM"
                plt.errorbar(time[keep], lc[keep], yerr=elc[keep], fmt='k.', alpha=0.8,
                             label=r"{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comp_label, mlc, rms*1000))
                # plt.plot(time[keep], lc[keep], "k-", lw=0.5)

            offset = np.nanpercentile(lc[keep], 1.0) - 4.0*rms

            plt.hlines(offset, mintime, maxtime,
                       colors='k', linestyle='-', lw=0.5)
            plt.hlines(offset-rms, mintime, maxtime,
                       colors='k', linestyle='--', lw=0.5)
            plt.hlines(offset+rms, mintime, maxtime,
                       colors='k', linestyle='--', lw=0.5)

        else:
            if plot_comps:
                comp_label = "C{:03d}".format(comps[i-1])
                plt.errorbar(time[keep], (lc[keep]-mlc)+offset, yerr=elc[keep], fmt='.', alpha=0.5,
                             label=r"{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comp_label, mlc, rms*1000))

        # print(i, mlc, rms)

    plt.xlabel(r"time (BJD)", fontsize=16)
    plt.ylabel(r"$\Delta$mag", fontsize=16)
    plt.legend(fontsize=10)
    
    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()
        

def plot_light_curve(filename, target=0, comps=[], nsig=10,
                     plot_coords=True, plot_rawmags=True, plot_sum=True,
                     plot_comps=True, catalog_name=1, output="",
                     output_coords="", output_rawmags="", output_lc="",figsize=(12, 6)):
    """ Plot light curve

    Parameters
    ----------
    filename : str
        string for fits file path

    output : str, optional
        The output csv table file name to save differential photometry data. If empty, it won't be saved.
        
    output_coords : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.
    output_rawmags : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.
    output_lc : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    results : dict
        dict container with light curve data
    """
    
    # initialize results dict container
    results = Table()
    
    # open time series fits file
    hdul = fits.open(filename)

    # get big table with data
    tbl = hdul[catalog_name].data

    # get data target data
    target_tbl = tbl[tbl["SRCINDEX"] == target]

    time = target_tbl['TIME']
    mintime, maxtime = np.min(time), np.max(time)

    x = target_tbl['X']
    y = target_tbl['Y']
    fwhm = target_tbl['FWHM']

    if plot_coords:
        fig = plt.figure(figsize=figsize)
        use_sky_coords = True
        platescale, unit = 1.0, 'pix'
        if use_sky_coords:
            platescale, unit = 0.335, 'arcsec'
        
        pixoffset = 7*platescale
        # fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [1, 1, 1, 1]})
        plt.plot(time, (x-np.nanmedian(x))*platescale + pixoffset, '.', color='darkblue', label='x-offset')
        plt.plot(time, (y-np.nanmedian(y))*platescale - pixoffset, '.', color='brown', label='y-offset')
        mfhwm = np.nanmedian(fwhm)
        plt.plot(time, (fwhm-mfhwm)*platescale, '.', color='darkgreen', label='FWHM - median={:.1f} {}'.format(mfhwm*platescale, unit))
        plt.xlabel(r"time (BJD)", fontsize=16)
        plt.ylabel(r"$\Delta$ {}".format(unit), fontsize=16)
        plt.legend(fontsize=10)
        if output_coords != '' :
            fig.savefig(output_coords, bbox_inches='tight')
            plt.close(fig)
        else :
            plt.show()

    mag_offset = 0.1
    m = target_tbl['MAG']
    em = target_tbl['EMAG']
    skym = target_tbl['SKYMAG']
    eskym = target_tbl['ESKYMAG']
    mmag = np.nanmedian(m)
    mskym = np.nanmedian(skym)

    if plot_rawmags:
        fig = plt.figure(figsize=figsize)
        plt.errorbar(time, mmag - m - mag_offset, yerr=em, fmt='.', label='raw obj dmag, mean={:.4f}'.format(mmag))
        plt.plot(time, mskym - skym + mag_offset, '.', label='raw sky dmag, mean={:.4f}'.format(mskym))
        plt.xlabel(r"time (BJD)", fontsize=16)
        plt.ylabel(r"$\Delta$mag", fontsize=16)
        plt.legend(fontsize=10)
        if output_rawmags != '' :
            fig.savefig(output_rawmags, bbox_inches='tight')
            plt.close(fig)
        else :
            plt.show()
            
    results['TIME'] = time
    results['x'] = x
    results['y'] = y
    results['fwhm'] = fwhm
    results['mag'] = m
    results['mag_err'] = em
    results['skymag'] = skym
    results['skymag_err'] = eskym
    
    cm, ecm = [], []
    sumag = np.zeros_like(m)
    sumag_var = np.zeros_like(m)
    
    if plot_comps or plot_sum:
        fig = plt.figure(figsize=figsize)
    
    for i in range(len(comps)):
        comp_tbl = tbl[tbl["SRCINDEX"] == comps[i]]

        cmag = comp_tbl['MAG']
        ecmag = comp_tbl['EMAG']
        cm.append(cmag)
        ecm.append(ecmag)

        sumag += 10**(-0.4*cmag)
        sumag_var += ecmag*ecmag
        
        mdm = np.nanmedian(cmag - m)
        dm = (cmag - m) - mdm
        
        rms = np.nanmedian(np.abs(dm)) / 0.67449
        edm = np.sqrt(ecmag**2 + em**2)
        
        results['diffmag_C{:05d}'.format(i)] = dm
        results['diffmag_err_C{:05d}'.format(i)] = edm

        keep = np.isfinite(dm)
        keep &= np.abs(dm) < nsig*rms

        if plot_comps:
            plt.errorbar(time[keep], dm[keep], yerr=edm[keep], fmt='.', alpha=0.3,
                         label=r"C{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comps[i], mdm, rms*1000))

    sumag = -2.5*np.log10(sumag)
    mdm = np.nanmedian(sumag - m)
    dm = (sumag - m) - mdm
    sumag_err = np.sqrt(sumag_var)
    
    results['magsum'] = sumag
    results['magsum_err'] = sumag_err
    results['diffmagsum'] = dm

    if plot_sum:
        
        rms = np.nanmedian(np.abs(dm)) / 0.67449
        keep = np.isfinite(dm)
        keep &= np.abs(dm) < nsig*rms
        
        plt.errorbar(time[keep], dm[keep], yerr=em[keep], fmt='k.',
                     label=r"SUM $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(mdm, rms*1000))

    if plot_comps or plot_sum:
        plt.xlabel(r"time (BJD)", fontsize=16)
        plt.ylabel(r"$\Delta$mag", fontsize=16)
        plt.legend(fontsize=10)
        if output_lc != '' :
            fig.savefig(output_lc, bbox_inches='tight')
            plt.close(fig)
        else :
            plt.show()

    if output != "" :
        # save output table to file
        results.write(output, overwrite=True)

    return results
    

def plot_polarimetry_results(loc, pos_model_sampling=1, title_label="", wave_plate='halfwave', output='',figsize=(12, 6)):
    """ Pipeline module to plot half-wave polarimetry data

    Parameters
    ----------
    loc : dict
        container with polarimetry data results
    pos_model_sampling : int
        step size for sampling of the position angle (deg) in the polarization model
    title_label : str
        plot title
    wave_plate : str
        wave plate mode

    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    waveplate_angles = loc["WAVEPLATE_ANGLES"]
    zi = loc["ZI"]
    qpol = loc["Q"]
    upol = loc["U"]
    vpol = loc["V"]
    ppol = loc["P"]
    theta = loc["THETA"]
    kcte = loc["K"]
    zero = loc["ZERO"]
    tsigma = loc["TSIGMA"]
    
    tsigmalab = r"$\sigma_t$: {:.5f} %".format(100*tsigma)
    kctelab = r"k: {:.5f}".format(kcte.nominal)
    zerolab = r"$\psi_0$: {} deg".format(zero)
    title_label += "  " + tsigmalab + "  " + kctelab + "  " + zerolab

    qlab = r"q: {:.4f}$\pm${:.4f} %".format(100*qpol.nominal, 100*qpol.std_dev)
    ulab = r"u: {:.4f}$\pm${:.4f} %".format(100*upol.nominal, 100*upol.std_dev)
    vlab = ""
    if wave_plate == 'quarterwave' :
        vlab = r"v: {:.4f}$\pm${:.4f} %".format(100*vpol.nominal, 100*vpol.std_dev)
    plab = r"p: {:.4f}$\pm${:.4f} %".format(100*ppol.nominal, 100*ppol.std_dev)
    thetalab = r"$\theta$: {:.2f}$\pm${:.2f} deg".format(theta.nominal, theta.std_dev)
        
    title_label += "\n"+qlab+"  "+ulab
    if wave_plate == 'quarterwave':
        title_label += "  "+vlab
    title_label += "  "+plab+"  "+thetalab

    # plot best polarimetry results
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})

    if title_label != "":
        axes[0].set_title(title_label)

    # define grid of position angle points for model
    pos_model = np.arange(0, 360, pos_model_sampling)

    # Plot the model
    best_fit_model = np.full_like(pos_model, np.nan)
    if wave_plate == 'halfwave':
        best_fit_model = halfwave_model(pos_model, qpol.nominal, upol.nominal)
    elif wave_plate == 'quarterwave':
        best_fit_model = quarterwave_model(pos_model, qpol.nominal, upol.nominal, vpol.nominal, zero=zero.nominal)

    axes[0].plot(pos_model, best_fit_model, 'r:', alpha=0.8, label='Best fit model')
    # axes[0].fill_between(pos_model, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3, edgecolor="none")

    # Plot data
    axes[0].errorbar(waveplate_angles, zi.nominal, yerr=zi.std_dev, fmt='ko', ms=2, capsize=2, lw=0.5, alpha=0.9, label='data')
    axes[0].set_ylabel(r"$Z(\phi) = \frac{f_\parallel(\phi)-f_\perp(\phi)}{f_\parallel(\phi)+f_\perp(\phi)}$", fontsize=16)
    axes[0].legend(fontsize=16)
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].minorticks_on()
    axes[0].tick_params(which='minor', length=3, width=0.7,direction='in', bottom=True, top=True, left=True, right=True)
    axes[0].tick_params(which='major', length=7, width=1.2,direction='in', bottom=True, top=True, left=True, right=True)

    # Print q, u, p and theta values
    ylims = axes[0].get_ylim()
    # axes[0].text(-10, ylims[1]-0.06,'{}\n{}\n{}\n{}'.format(qlab, ulab, plab, thetalab), size=12)

    # Plot residuals
    observed_model = np.full_like(waveplate_angles, np.nan)
    if wave_plate == 'halfwave':
        observed_model = halfwave_model(waveplate_angles, qpol.nominal, upol.nominal)
    elif wave_plate == 'quarterwave':
        observed_model = quarterwave_model(waveplate_angles, qpol.nominal, upol.nominal, vpol.nominal, zero=zero.nominal)

    resids = zi.nominal - observed_model
    sig_res = np.nanstd(resids)

    axes[1].errorbar(waveplate_angles, resids, yerr=zi.std_dev,fmt='ko', alpha=0.5, label='residuals')
    axes[1].set_xlabel(r"waveplate position angle, $\phi$ [deg]", fontsize=16)
    axes[1].hlines(0., 0, 360, color="k", linestyles=":", lw=0.6)
    axes[1].set_ylim(-5*sig_res, +5*sig_res)
    axes[1].set_ylabel(r"residuals", fontsize=16)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].minorticks_on()
    axes[1].tick_params(which='minor', length=3, width=0.7,direction='in', bottom=True, top=True, left=True, right=True)
    axes[1].tick_params(which='major', length=7, width=1.2,direction='in', bottom=True, top=True, left=True, right=True)

    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()


def plot_2d(x, y, z, LIM=None, LAB=None, z_lim=None, use_index_in_y=False, title="", pfilename="", cmap="gist_heat"):
    """
    Use pcolor to display sequence of spectra

    Inputs:
    - x:        x array of the 2D map (if x is 1D vector, then meshgrid; else: creation of Y)
    - y:        y 1D vector of the map
    - z:        2D array (sequence of spectra; shape: (len(x),len(y)))
    - LIM:      list containing: [[lim_inf(x),lim_sup(x)],[lim_inf(y),lim_sup(y)],[lim_inf(z),lim_sup(z)]]
    - LAB:      list containing: [label(x),label(y),label(z)] - label(z) -> colorbar
    - title:    title of the map
    - **kwargs: **kwargs of the matplolib function pcolor

    Outputs:
    - Display 2D map of the sequence of spectra z
    None
    """

    if use_index_in_y:
        y = np.arange(len(y))

    if len(np.shape(x)) == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X = x
        Y = []
        for n in range(len(x)):
            Y.append(y[n] * np.ones(len(x[n])))
        Y = np.array(Y, dtype=float)
    Z = z

    if LIM == None:
        x_lim = [np.min(X), np.max(X)]  # Limits of x axis
        y_lim = [np.min(Y), np.max(Y)]  # Limits of y axis
        if z_lim == None:
            z_lim = [np.min(Z), np.max(Z)]
        LIM = [x_lim, y_lim, z_lim]

    if LAB == None:
        # Labels of the map
        x_lab = r"$Velocity$ [km/s]"  # Wavelength axis
        y_lab = r"Time [BJD]"  # Time axis
        z_lab = r"CCF"  # Intensity (exposures)
        LAB = [x_lab, y_lab, z_lab]

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10, 7)
    ax = plt.subplot(111)

    cc = ax.pcolor(X, Y, Z, vmin=LIM[2][0], vmax=LIM[2][1], cmap=cmap)
    cb = plt.colorbar(cc, ax=ax)

    ax.set_xlim(LIM[0][0], LIM[0][1])
    ax.set_ylim(LIM[1][0], LIM[1][1])

    ax.set_xlabel(LAB[0], fontsize=20)
    ax.set_ylabel(LAB[1], labelpad=15, fontsize=20)
    cb.set_label(LAB[2], rotation=270, labelpad=30, fontsize=20)

    ax.set_title(title, pad=15, fontsize=20)

    if pfilename == "":
        plt.show()
    else:
        plt.savefig(pfilename, format='png')
    plt.clf()
    plt.close()


def plot_polar_time_series(filename, target=0, comps=[], nsig=10, plot_total_polarization=True, plot_polarization_angle=True, plot_comps=True, plot_sum=True, plot=True, output="",figsize=(12, 6)) :

    """ Tool to plot polarimetric time series

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

    # initialize results dict container
    results = Table()
    
    # open time series fits file
    hdul = fits.open(filename)

    # get polarimetry type "halfwave" or "quarterwave"
    poltype = hdul[0].header["POLTYPE"]

    # get big table with data
    tbl = hdul["POLAR_TIMESERIES"].data
    
    """
    #Below are the contents of extension "POLAR_TIMESERIES" in polarimetric ts FITS product
    dtype=(numpy.record, [('TIME', '>f8'), ('SRCINDEX', '>f8'), ('RA', '>f8'), ('DEC', '>f8'), ('X1', '>f8'), ('Y1', '>f8'), ('X2', '>f8'), ('Y2', '>f8'), ('FWHM', '>f8'), ('MAG', '>f8'), ('EMAG', '>f8'), ('Q', '>f8'), ('EQ', '>f8'), ('U', '>f8'), ('EU', '>f8'), ('V', '>f8'), ('EV', '>f8'), ('P', '>f8'), ('EP', '>f8'), ('THETA', '>f8'), ('ETHETA', '>f8'), ('K', '>f8'), ('EK', '>f8'), ('ZERO', '>f8'), ('EZERO', '>f8'), ('NOBS', '>f8'), ('NPAR', '>f8'), ('CHI2', '>f8')]))
    """
    
    # get data target data
    target_tbl = tbl[tbl["SRCINDEX"] == target]

    # set time axis
    time = target_tbl['TIME']
    mintime, maxtime = np.min(time), np.max(time)

    x1 = target_tbl['X1']
    y1 = target_tbl['Y1']
    x2 = target_tbl['X2']
    y2 = target_tbl['Y2']
    fwhm = target_tbl['FWHM']
    mag_offset = 0.1

    # set minimum number of rows: 1 for photometry and 1 for polarimetry
    nrows = 2

    # add a row to plot circular polarization
    if poltype == "quarterwave" :
        nrows += 1
        
    if plot_total_polarization :
        # for total polarization, add a row for theta if requested
        if plot_polarization_angle :
            nrows += 1
        pass
    else :
        # for linear components of polarization (Q and U) we need one more row
        nrows += 1
        
    if plot :
        # set plot frame
        fig, axs = plt.subplots(nrows, figsize=(12, 6), sharex=True, sharey=False)
       
    ##############################
    # START PLOTTING PHOTOMETRY  #
    ##############################
    axindex = 0
    
    # FIRST CALCULATE PHOTOMETRY :
    mag_offset = 0.1
    m = target_tbl['MAG']
    em = target_tbl['EMAG']
    mmag = np.nanmedian(m)
    
    cm, ecm = [], []
    sumag = np.zeros_like(m)
    sumag_var = np.zeros_like(m)

    for i in range(len(comps)):
        comp_tbl = tbl[tbl["SRCINDEX"] == comps[i]]

        cmag = comp_tbl['MAG']
        ecmag = comp_tbl['EMAG']
        cm.append(cmag)
        ecm.append(ecmag)

        sumag += 10**(-0.4*cmag)
        sumag_var += ecmag*ecmag

        mdm = np.nanmedian(cmag - m)
        dm = (cmag - m) - mdm
        rms = np.nanmedian(np.abs(dm)) / 0.67449
        edm = np.sqrt(ecmag**2 + em**2)
        keep = np.isfinite(dm)
        keep &= np.abs(dm) < nsig*rms

        results['diffmag_C{:05d}'.format(i)] = dm
        results['diffmag_err_C{:05d}'.format(i)] = edm

        if plot_comps and plot:
            axs[axindex].errorbar(time[keep], dm[keep], yerr=edm[keep], fmt='.', alpha=0.3,label=r"C{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comps[i], mdm, rms*1000))

    sumag = -2.5*np.log10(sumag)
    mdm = np.nanmedian(sumag - m)
    dm = (sumag - m) - mdm
    sumag_err = np.sqrt(sumag_var)
    
    rms = np.nanmedian(np.abs(dm)) / 0.67449
    keep = np.isfinite(dm)
    # keep &= np.abs(dm) < nsig*rms
    
    if plot_sum and plot:
        axs[axindex].errorbar(time[keep], dm[keep], yerr=em[keep], fmt='k.', label=r"SUM $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(mdm, rms*1000))

    if plot and (plot_comps or plot_sum) :
        axs[axindex].set_ylabel(r"$\Delta$mag", fontsize=16)
        axs[axindex].tick_params(axis='y', labelsize=14)
        axs[axindex].legend(fontsize=10)
        axs[axindex].minorticks_on()
        axs[axindex].tick_params(which='minor', length=3, width=0.7, direction='in', bottom=True, top=True, left=True, right=True)
        axs[axindex].tick_params(which='major', length=7, width=1.2,direction='in', bottom=True, top=True, left=True, right=True)
    
    results['TIME'] = time
    results['x1'] = x1
    results['y1'] = y1
    results['x2'] = x2
    results['y2'] = y2
    results['fwhm'] = fwhm
    results['mag'] = m
    results['mag_err'] = em
    
    results['magsum'] = sumag
    results['magsum_err'] = sumag_err
    results['diffmagsum'] = dm

    ##############################
    # START PLOTTING POLARIMETRY #
    ##############################
    
    # PLOT CIRCULAR POLARIZATION IF L4
    if poltype == "quarterwave" :
        axindex += 1
        circ_polarization = target_tbl['V']
        circ_polarization_error = target_tbl['EV']

        cpmedian = np.nanmedian(circ_polarization)
        cprms = np.nanmedian(np.abs(circ_polarization - cpmedian)) / 0.67449
    
        if plot :
            axs[axindex].errorbar(time, circ_polarization*100, yerr=circ_polarization_error*100, fmt='k.', label=r"V[{}] = {:.3f}$\pm${:.3f} %".format(target, cpmedian*100, cprms*100))
            axs[axindex].set_ylabel(r"V (%)", fontsize=16)
            axs[axindex].tick_params(axis='y', labelsize=14)
            axs[axindex].legend(fontsize=10)
            axs[axindex].minorticks_on()
            axs[axindex].tick_params(which='minor', length=3, width=0.7, direction='in', bottom=True, top=True, left=True, right=True)
            axs[axindex].tick_params(which='major', length=7, width=1.2,direction='in', bottom=True, top=True, left=True, right=True)
    
    if plot_total_polarization :
        plot1_key='P'
        plot1_error_key='EP'
        plot2_key='THETA'
        plot2_error_key='ETHETA'
    else :
        plot1_key='Q'
        plot1_error_key='EQ'
        plot2_key='U'
        plot2_error_key='EU'
    
    # PLOT first polarization component (P or Q)
    axindex += 1
    
    polarization = target_tbl[plot1_key]
    polarization_error = target_tbl[plot1_error_key]

    pmedian = np.nanmedian(polarization)
    prms = np.nanmedian(np.abs(polarization - pmedian)) / 0.67449
    if plot :
        axs[axindex].errorbar(time, polarization*100, yerr=polarization_error*100, fmt='k.', label=r"{}[{}] = {:.3f}$\pm${:.3f} %".format(plot1_key,target,pmedian*100, prms*100))
        axs[axindex].set_ylabel(r"{} (%)".format(plot1_key), fontsize=16)
        axs[axindex].tick_params(axis='y', labelsize=14)
        axs[axindex].legend(fontsize=10)
        axs[axindex].minorticks_on()
        axs[axindex].tick_params(which='minor', length=3, width=0.7, direction='in', bottom=True, top=True, left=True, right=True)
        axs[axindex].tick_params(which='major', length=7, width=1.2,direction='in', bottom=True, top=True, left=True, right=True)
    
    results['polarization_1'] = np.array(polarization,dtype=float)
    results['polarization_1_err'] = np.array(polarization_error,dtype=float)
    
    # PLOT second polarization component (THETA or U)
    y2 = target_tbl[plot2_key]
    y2err = target_tbl[plot2_error_key]
    
    y2median = np.nanmedian(y2)
    y2rms = np.nanmedian(np.abs(y2 - y2median)) / 0.67449
    
    if plot_total_polarization :
        if plot_polarization_angle :
            # plot polarization angle (theta)
            axindex += 1
            if plot :
                axs[axindex].errorbar(time, y2, yerr=y2err, fmt='k.', label=r"$\theta$[{}] = {:.1f}$\pm${:.1f} deg".format(target, y2median, y2rms))
                axs[axindex].set_ylabel(r"$\theta$ (deg)", fontsize=16)
    else :
        # plot U
        axindex += 1
        if plot :
            axs[axindex].errorbar(time, y2*100, yerr=y2err*100, fmt='k.', label=r"{}[{}] = {:.3f}$\pm${:.3f} %".format(plot2_key,target,y2median*100, y2rms*100))
            axs[axindex].set_ylabel(r"{} (%)".format(plot2_key), fontsize=16)

    results['polarization_2'] = np.array(y2,dtype=float)
    results['polarization_2_err'] = np.array(y2err,dtype=float)
    
    if plot :
        axs[axindex].legend(fontsize=10)
        axs[axindex].tick_params(axis='x', labelsize=14)
        axs[axindex].tick_params(axis='y', labelsize=14)
        axs[axindex].minorticks_on()
        axs[axindex].set_xlabel(r"time (BJD)", fontsize=16)
        axs[axindex].tick_params(which='minor', length=3, width=0.7, direction='in', bottom=True, top=True, left=True, right=True)
        axs[axindex].tick_params(which='major', length=7, width=1.2, direction='in', bottom=True, top=True, left=True, right=True)
        if output != '' :
            fig.savefig(output, bbox_inches='tight')
            plt.close(fig)
        else :
            plt.show()

    
    return results


def plot_polarimetry_map(stack_product, polar_product, min_aperture=0, max_aperture=1024, percentile=99.5,ref_catalog="CATALOG_POL_N_AP010", src_label_offset=30, arrow_size_scale=None, compute_k=False, title_label="", output="",figsize=(10, 10)):
    """ Pipeline module to plot polarimetry map

    Parameters
    ----------
    stack_product : str
        path to stack image product
    polar_product : str
        path to polarimetry product
        
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    hdul = fits.open(stack_product)
    img_data = hdul["PRIMARY"].data
    x_o, y_o = hdul["CATALOG_POL_N_AP010"].data['x'], hdul["CATALOG_POL_N_AP010"].data['y']
    x_e, y_e = hdul["CATALOG_POL_S_AP010"].data['x'], hdul["CATALOG_POL_S_AP010"].data['y']

    hdr = fits.getheader(polar_product)
    nsources = hdr["NSOURCES"]

    pol, epol = [], []
    pa, epa = [], []
    q ,u = [], []

    for i in range(nsources) :
        pol_results = s4pipelib.get_polarimetry_results(polar_product,
                                                        source_index=i,
                                                        min_aperture=min_aperture,
                                                        max_aperture=max_aperture,
                                                        compute_k=compute_k,
                                                        plot=False,
                                                        verbose=False)

        pol.append(pol_results["P"].nominal*100)
        epol.append(pol_results["P"].std_dev*100)
        pa.append(pol_results["THETA"].nominal)
        epa.append(pol_results["THETA"].std_dev)
    
        q.append(pol_results["Q"].nominal*100)
        u.append(pol_results["U"].nominal*100)
    
    q = np.array(q)
    u = np.array(u)

    src_idxs = np.arange(nsources)

    fig = plt.figure(figsize=figsize)
    plt.imshow(img_data, vmin=np.percentile(img_data, 100-percentile), vmax=np.percentile(img_data, percentile), origin='lower')
    
    mean_aper = np.mean(hdul[ref_catalog].data['APER'])
    plt.plot(x_o, y_o, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
    plt.plot(x_e, y_e, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)

    
    if arrow_size_scale is None :
        # calculate scale to define size of arrows
        totp = np.sqrt(q**2+u**2)
        arrow_size_scale = (np.nanmax(totp) + np.nanmin(totp))/(2*100)  # average between maximum and minimum divided by 100
    
    plt.quiver(x_o, y_o, -q, u, color='w', units='xy', scale=arrow_size_scale, width=4, headwidth=4, headlength=4, headaxislength=3, alpha=1, zorder=2)

    for i in range(len(x_o)):
        x = [x_o[i], x_e[i]]
        y = [y_o[i], y_e[i]]
        plt.plot(x, y, '-o', color="gray", zorder=1)
        plt.annotate(f"{i}", [np.mean(x)-src_label_offset, np.mean(y)+src_label_offset], color="gray",zorder=3)
    
    if title_label != "":
        plt.title(title_label)
    
    plt.xlabel("col [pix]")
    plt.ylabel("row [pix]")
    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()


def plot_s4_lightcurves(lcfiles=[None,None,None,None], object_indexes=[0,0,0,0], comps=[[1],[1],[1],[1]], aperture_radius=10, object_name='', output='',figsize=(10, 10)) :

    """ Module to plot differential light curves for all 4 channels
    
    Parameters
    ----------
    lcfiles: [str,str,str,str]
        list of light curves file paths for all 4 channels
    object_indexes: list: [int, int, int, int]
        list of target indexes
    comps: list: [list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..]]
        list of arrays of comparison indexes in the four channels
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    object_name : str
        object name
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
        lcs: list
            set of 4 lightcurve data
    -------
    """
    
    bands = ['g', 'r', 'i', 'z']
    colors = ['darkblue','darkgreen','darkorange','darkred']
    
    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=figsize)

    lcs = []
    
    for ch in range(4) :
        if ch == 0 :
            axs[ch].set_title(r"{}".format(object_name), fontsize=20)
        try :
            lc = plot_light_curve(lcfiles[ch], target=object_indexes[ch], comps=comps[ch], catalog_name=catalog_ext, plot_coords=False, plot_rawmags=False, plot_sum=False, plot_comps=False)
            axs[ch].plot(lc['TIME'],lc['diffmagsum'],".",color=colors[ch], alpha=0.3, label=bands[ch])
            lcs.append(lc)
        except :
            lcs.append(None)
            continue
                
        axs[ch].tick_params(axis='x', labelsize=14)
        axs[ch].tick_params(axis='y', labelsize=14)
        axs[ch].minorticks_on()
        axs[ch].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs[ch].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
        if ch == 3 :
             axs[ch].set_xlabel(r"time (BJD)", fontsize=16)
                
        axs[ch].set_ylabel(r"$\Delta$mag", fontsize=16)
        axs[ch].legend(fontsize=16)

    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()
    
    return lcs


def plot_s4data_for_diffphot(sci_img_files=[None,None,None,None], object_indexes=[0,0,0,0], comps=[[1],[1],[1],[1]], instmode="PHOT", percentile=98, aperture_radius=10, object_name="", fontsize=10, output='',figsize=(10, 10)) :


    """ Module to plot differential light curves and auxiliary data
    
    Parameters
    ----------
    sci_img_files: [str,str,str,str]
        list of processed scientific image file paths for all 4 channels
    object_indexes: list: [int, int, int, int]
        list of target indexes
    comps: list: [list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..]]
        list of arrays of comparison indexes in the four channels
    instmode : str
        instrument mode: "PHOT" or "POLAR"
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    object_name : str
        object name
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
        
    -------
    """

    nstars = 1

    bands = ['g', 'r', 'i', 'z']
    colors = ['lightblue','darkgreen','darkorange','red']
    panel_pos = [[0,0],[0,1],[1,0],[1,1]]
    
    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
    if instmode == "POLAR" :
        catalog_ext = "CATALOG_POL_N_AP{:03d}".format(aperture_radius)

    nrows, ncols = 2, 2
    #fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, layout='constrained', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=figsize)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        
    for ch in range(4) :
        col, row = panel_pos[ch]
        
        axs[col,row].set_title(r"{}-band".format(bands[ch]),fontsize=fontsize)

        if sci_img_files[ch] is not None :
            
            hdul = fits.open(sci_img_files[ch])
            img_data, hdr = hdul[0].data, hdul[0].header
            catalog = Table(hdul[catalog_ext].data)
            wcs = WCS(hdr, naxis=2)
        
            for j in range(len(catalog["X"])) :
                
                #axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius+2, facecolor="none", edgecolor=colors[ch]))
                
                if j == object_indexes[ch] :
                    axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j]-1.1*aperture_radius, '{}'.format(j), c='gray', fontsize=fontsize)
                    
                    axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], '{}'.format(object_name), c='white', fontsize=fontsize, alpha=0.5)
                    axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor=colors[ch]))
                    axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius+2, facecolor="none", edgecolor=colors[ch]))
                else :
                    if j in comps[ch] :
                        axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], 'C{}'.format(j), c='white', fontsize=fontsize)
                        axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor=colors[ch]))
                    else :
                        axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], '{}'.format(j), c='gray', fontsize=fontsize)
                        axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor="grey"))
                            
        else :
            img_data = np.empty([1024, 1024])
  
        axs[col,row].imshow(img_data, vmin=np.percentile(img_data, 100. - percentile), vmax=np.percentile(img_data, percentile), origin='lower', cmap='cividis', aspect='equal')
                  
        axs[col,row].tick_params(axis='x', labelsize=fontsize)
        axs[col,row].tick_params(axis='y', labelsize=fontsize)
        axs[col,row].minorticks_on()
        axs[col,row].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs[col,row].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
                         
    if output != '' :
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()

    return

