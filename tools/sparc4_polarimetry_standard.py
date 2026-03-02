"""
    Created on Nov 3 2025
    
    Description: A tool to analyze a polarimetric standard star observed with SPARC4
    
    @author: Eder Martioli <emartioli@lna.br>
    Laboratório Nacional de Astrofísica, Brasil.

    Simple usage examples:
    
    python sparc4_polarimetry_standard.py --object="Hilt 652" --input="/Users/eder/Science/paper_sparc4-pipeline/PolarimetricCalibration/Hilt652/" --obj_indexes="0,0,0,0" --ref_pol="5.948±0.017,6.371±0.009,6.218±0.004,5.613±0.004" --ref_theta="179.52±0.05,179.44±0.03,179.39±0.03,179.46±0.03" --ref_bands="B,V,R,I" -pv
    
    python sparc4_polarimetry_standard.py --object="Hilt 715" --input="/Users/eder/Science/paper_sparc4-pipeline/PolarimetricCalibration/Hilt715/" --obj_indexes="1,2,4,3" --ref_pol="5.801±0.024,6.10±0.05,5.818±0.006,4.99±0.006" --ref_theta="49.70±0.07,49.8±0.05,49.7±0.04,49.44±0.05" --ref_bands="B,V,R,I" -pv
    
    python sparc4_polarimetry_standard.py --object="HD 187929" --input="/Users/eder/Science/paper_sparc4-pipeline/PolarimetricCalibration/HD187929/" --obj_indexes="0,0,0,0" --ref_pol="1.65±0.02,1.67±0.02" --ref_theta="94.4±0.4,93.3±0.3" --ref_bands="B,R"
    
    
    python sparc4_polarimetry_standard.py --object="Hilt 652" --input="/Users/eder/Science/paper_sparc4-pipeline/PolarimetricCalibration/Hilt652/" --obj_indexes="0,0,0,0" --ref_pol="5.948±0.017,5.7±0.01,5.8±0.01,6.371±0.009,6.25±0.03,6.32±0.01,6.218±0.004,6.07±0.02,5.613±0.004,5.61±0.04" --ref_theta="179.52±0.05,179.47±0.13,179.85±0.05,179.44±0.03,179.18±0.2,179.27±0.07,179.39±0.03,179.39±0.1,179.46±0.03,179.18±0.11" --ref_bands="B,B,B,V,V,V,R,R,I,I" -pv
    
    python sparc4_polarimetry_standard.py --object="Hilt 715" --input="/Users/eder/Science/paper_sparc4-pipeline/PolarimetricCalibration/Hilt715/" --obj_indexes="1,2,4,3" --ref_pol="5.801±0.024,5.71±0.01,5.74±0.01,6.10±0.05,6.07±0.02,6.06±0.01,5.818±0.006,5.69±0.07,4.99±0.006,5.07±0.01" --ref_theta="49.70±0.07,49.93±0.15,49.88±0.07,49.8±0.05,49.62 ± 0.08,49.35±0.03,49.7±0.04,49.90±0.24,49.44±0.05,48.92±0.19" --ref_bands="B,B,B,V,V,V,R,R,I,I"
    
    """
    

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys, os
from optparse import OptionParser

import sparc4.pipeline_lib as s4pipelib
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table
import glob

from uncertainties import ufloat,umath

from scipy import optimize
import emcee
import corner
import matplotlib
from copy import deepcopy


def load_data_products(datadir, object_id=None, pol_beam='S+N') :
    """ Module to identify and store the paths of s4 data products within a given directory
    Parameters
    ----------
    datadir : path
        directory path to search products
    obj_id : str
        object identification name
    Returns
        loc : dict
            dictionary data container to store product paths.
            The keys for each entry follow the structure objectname_instmode
    -------
    """

    loc = {}

    # get list of stack products
    stackfiles = sorted(glob.glob("{}/*stack.fits".format(datadir)))
    
    # get instrument mode (PHOT or POLAR) from stack images that match object id
    objs_istmode = []
    for i in range(len(stackfiles)) :
        hdr = fits.getheader(stackfiles[i])
        if hdr["OBJECT"] not in objs_istmode and (object_id is None or object_id.replace(" ","").upper() == hdr["OBJECT"].replace(" ","").upper()) :
            objs_istmode.append([hdr["OBJECT"],hdr["INSTMODE"]])

    # for each object / instrument mode find all other products
    for obj,instmode in objs_istmode :
        key = "{}_{}".format(obj,instmode)
    
        loc[key] = {}
        
        loc[key]["object"] = obj
        loc[key]["instmode"] = instmode

        loc[key]["stack"] = [None,None,None,None]
        loc[key]["ts"] = [None,None,None,None]
        loc[key]["lc"] = [None,None,None,None]
        loc[key]["polarl2"] = [None,None,None,None]
        loc[key]["polarl4"] = [None,None,None,None]

        for i in range(len(stackfiles)) :
            hdr = fits.getheader(stackfiles[i])
            if loc[key]["stack"][hdr["CHANNEL"]-1] is None :
                loc[key]["stack"][hdr["CHANNEL"]-1] = stackfiles[i]
            
        if instmode == "POLAR":
        
            lcfiles = sorted(glob.glob("{}/*{}_lc.fits".format(datadir,pol_beam)))
           
            tsfiles = sorted(glob.glob("{}/*ts.fits".format(datadir)))
            polarl2files = sorted(glob.glob("{}/*_POLAR_L2*polar.fits".format(datadir)))
            polarl4files = sorted(glob.glob("{}/*_POLAR_L4*polar.fits".format(datadir)))

            for i in range(len(tsfiles)) :
                hdr = fits.getheader(tsfiles[i])
                if loc[key]["ts"][hdr["CHANNEL"]-1] is None :
                    loc[key]["ts"][hdr["CHANNEL"]-1] = tsfiles[i]
                    
            for i in range(len(polarl2files)) :
                hdr = fits.getheader(polarl2files[i])
                if loc[key]["polarl2"][hdr["CHANNEL"]-1] is None :
                    loc[key]["polarl2"][hdr["CHANNEL"]-1] = polarl2files[i]
                    
            for i in range(len(polarl4files)) :
                hdr = fits.getheader(polarl4files[i])
                if loc[key]["polarl4"][hdr["CHANNEL"]-1] is None :
                    loc[key]["polarl4"][hdr["CHANNEL"]-1] = polarl4files[i]
        else :
            lcfiles = sorted(glob.glob("{}/*lc.fits".format(datadir)))
        
        for i in range(len(lcfiles)) :
            hdr = fits.getheader(lcfiles[i])
            
            parts = lcfiles[i].split("_")
            
            for part in parts :
                if 's4c' in part :
                    ch = int(part[-1])
                    break
                    
            if "CHANNEL" in hdr.keys() :
                ch = hdr["CHANNEL"]
        
            if loc[key]["lc"][ch-1] is None :
                loc[key]["lc"][ch-1] = lcfiles[i]
                
    return loc


def serkowski(wl_max, p_max, k, wl) :
    p = p_max * np.exp(-k*np.log(wl_max/wl)**2)
    return p


def errfunc(params, wl, wlerr, p, perr, return_errors=False) :

    """ Function calculate the weighted residuals
    
    Parameters
    ----------
    wl_max: float
        maximum lambda
    k: float
        observed magnitudes
    wl: np.array()
        observed wavelengths
    wlerr: np.array()
        observed wavelength uncertainties
    p: np.array()
        observed polarizations
    perr: np.array()
        observed polarization uncertainties
    return_errors: np.array(), optional
        to return errror, useful for MCMC

    Returns
        residuals: np.array()
            residuals
        
    -------
    """
    
    wl_max, p_max, k = params[0], params[1], params[2]

    #errors = np.sqrt(perr*perr + wlerr*wlerr)
    errors = perr

    residuals = p - serkowski(wl_max, p_max, k, wl)

    if return_errors :
        return residuals/errors, errors
    else :
        return residuals/errors


def fit_Serkowski(ref_pol, run_mcmc=False) :
        
    pref, pref_err = ref_pol["POL"], ref_pol["EPOL"]
    
    wlref, wlref_err = ref_pol["WL"], ref_pol["EWL"]
    
    wl_max, p_max, k = 544., 6.0, 0.8
    pguess = np.array([wl_max, p_max, k])
    
    pfit, pcov, infodict, errmsg, success = optimize.leastsq(errfunc, pguess, args=(wlref, wlref_err, pref, pref_err), full_output=True)
    
    residuals = errfunc(pfit, wlref, wlref_err, pref, pref_err)

    errors = [None, None, None]
    if pcov is not None:
        pcov *= (residuals**2).sum()/(len(residuals)-len(pfit))
        for i in range(len(pfit)):
            errors.append(np.absolute(pcov[i][i])**0.5)

    if run_mcmc :
        pfit, errors, samples = run_mcmc_fit(pfit, wlref, wlref_err, pref, pref_err, amp=1e-4, nwalkers=50, niter=5000, burnin=2000, samples_filename="", appendsamples=False, verbose=True, plot=True)
        return pfit, errors, samples
    else :
        return pfit, errors, None


#- Derive best-fit params and their 1-sigm a error bars
def best_fit_params(samples, use_mean=False, best_fit_from_mode=False, nbins = 30, plot_distributions=False, use_mean_error=True, label_coeffs=None, verbose = False) :

    theta, theta_err = [], []
    
    if use_mean :
        npsamples = np.array(samples)
        values = []
        for i in range(len(samples[0])) :
            mean = np.mean(npsamples[:,i])
            err = np.std(npsamples[:,i])
            values.append([mean,err,err])
    else :
        func = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        percents = np.percentile(samples, [16, 50, 84], axis=0)
        seq = list(zip(*percents))
        values = list(map(func, seq))

        max_values = []
        
        for i in range(len(values)) :
            hist, bin_edges = np.histogram(samples[:,i], bins=nbins, range=(values[i][0]-5*values[i][1],values[i][0]+5*values[i][2]), density=True)
            xcen = (bin_edges[:-1] + bin_edges[1:])/2
            mode = xcen[np.argmax(hist)]
            max_values.append(mode)
            var_label = r"coeff_{}".format(i)
            if label_coeffs is not None :
                var_label = label_coeffs[i]
                    
            if plot_distributions :
                
                nmax = len(samples[:,i])
                plt.step(xcen, hist, where='mid')
                plt.vlines([values[i][0]], np.min(0), np.max(hist), ls="--", label="median")
                plt.vlines([mode], np.min(0), np.max(hist), ls=":", label="mode")
                plt.ylabel(r"Probability density",fontsize=18)
                plt.xlabel(var_label,fontsize=18)
                plt.legend()
                plt.show()

                plt.plot(samples[:,i],label=var_label, alpha=0.5, lw=0.5)
                plt.hlines([], np.min(0), np.max(nmax), ls=":", label="mode",zorder=2)
                plt.hlines([values[i][0]], np.min(0), np.max(nmax), ls="-", label="median",zorder=2)
                plt.ylabel(var_label,fontsize=18)
                plt.xlabel(r"MCMC iteration",fontsize=18)
                plt.legend(fontsize=18)
                plt.show()
                    
        max_values = np.array(max_values)

    for i in range(len(values)) :
        var_label = r"coeff_{}".format(i)
        if label_coeffs is not None :
            var_label = label_coeffs[i]
                
        if best_fit_from_mode :
            theta.append(max_values[i])
        else :
            theta.append(values[i][0])

        if use_mean_error :
            merr = (values[i][1]+values[i][2])/2
            theta_err.append(merr)
            if verbose :
                print("{} = {} +/- {}".format(var_label, values[i][0], merr))
        else :
            theta_err.append((values[i][1],values[i][2]))
            if verbose :
                print("{} = {} + {} - {}".format(var_label, values[i][0], values[i][1], values[i][2]))
                
    return theta, theta_err



#likelihood function
def lnlikelihood(coeffs, wl, wlerr, p, perr) :
    residuals, errors = errfunc(coeffs, wl, wlerr, p, perr, return_errors=True)
    sum_of_residuals = 0
    for i in range(len(residuals)) :
        sum_of_residuals += np.nansum(residuals[i]**2 + np.log(2.0 * np.pi * (perr[i] * perr[i]))  + np.log(2.0 * np.pi * (wlerr[i] * wlerr[i])) )
    ln_likelihood = -0.5 * (sum_of_residuals)
    return ln_likelihood

# prior probability from definitions in priorslib
def lnprior(theta):
    total_prior = 0.0
    """
    # below one may set bounds
    for i in range(len(theta)) :
        if -1e10 < theta[i] < +1e10:
            return 0
        return -np.inf
    """
    return total_prior

#posterior probability
def lnprob(theta, wl, wlerr, p, perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnlike = lnlikelihood(theta, wl, wlerr, p, perr)
    prob = lp + lnlike
    if np.isfinite(prob) :
        return prob
    else :
        return -np.inf
    
    
def run_mcmc_fit(params, wl, wlerr, p, perr, amp=1e-4, nwalkers=32, niter=100, burnin=20, samples_filename="", appendsamples=False, verbose=False, plot=False) :
    
    font = {'size': 12}
    matplotlib.rc('font', **font)
    
    theta = params
    if verbose:
        print("initializing emcee sampler ...")

    #amp, ndim, nwalkers, niter, burnin = 5e-4, len(theta), 50, 2000, 500
    ndim = len(theta)

    # Make sure the number of walkers is sufficient, and if not passing a new value
    if nwalkers < 2*ndim:
        nwalkers = 2*ndim
        print("WARNING: resetting number of MCMC walkers to {}".format(nwalkers))
    
    backend = None
    if samples_filename != "" :
        # Set up the backend
        backend = emcee.backends.HDFBackend(samples_filename)
        if appendsamples == False :
            backend.reset(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [wl, wlerr, p, perr], backend=backend)

    pos = [theta + amp * np.random.randn(ndim) for i in range(nwalkers)]
    #--------

    #- run mcmc
    if verbose:
        print("Running MCMC ...")
        print("N_walkers=",nwalkers," ndim=",ndim)

    sampler.run_mcmc(pos, niter, progress=True)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) # burnin : number of first samples to be discard as burn-in
    #--------

    theta_labels = [r"$\lambda_m$",r"p$_m$","k"]

    theta, theta_err = best_fit_params(samples, use_mean=False, best_fit_from_mode=True, nbins=30, plot_distributions=False, use_mean_error=True, label_coeffs=theta_labels, verbose=True)
    
    pfit = theta
    perr = theta_err
    
    if plot :
        fig = corner.corner(samples, labels=theta_labels, quantiles=[0.16, 0.5, 0.84], labelsize=8, show_titles=True, truths=pfit)
        plt.show()
    
    return pfit, perr, samples


def get_bandpasses () :

    bands = {}

    #bands['B'] = ufloat(440,50)
    #bands['V'] = ufloat(545,45)
    #bands['R'] = ufloat(649,80)
    #bands['I'] = ufloat(802,75)
    
    bands['B'] = ufloat(436.1,45)
    bands['V'] = ufloat(544.8,42)
    bands['R'] = ufloat(640.7,79)
    bands['I'] = ufloat(798,77)
    
    bands['g'] = ufloat(455.2,79.1)
    bands['r'] = ufloat(607.8,69.2)
    bands['i'] = ufloat(741.9,69.7)
    bands['z'] = ufloat(863.0,54)
    
    """
    bands['B'] = ufloat(445,0)
    bands['V'] = ufloat(551,0)
    bands['R'] = ufloat(658,0)
    bands['I'] = ufloat(806,0)
    
    bands['g'] = ufloat(455.2,0)
    bands['r'] = ufloat(607.8,0)
    bands['i'] = ufloat(741.9,0)
    bands['z'] = ufloat(863.0,0)
    """
    
    return bands


def get_ref_data(pol, theta, bands) :

    ref_pol = {}
    
    s_pol = pol.split(",")
    s_theta = theta.split(",")
    s_bands = bands.split(",")

    p, ep = [], []
    t, et = [], []
    wl, ewl = [], []
    
    for i in range(len(s_bands)) :
        uwl = get_bandpasses()[s_bands[i]]
    
        wl.append(uwl.nominal_value)
        ewl.append(uwl.std_dev)
        
        pref, epref = s_pol[i].split("±")
        thetaref, ethetaref = s_theta[i].split("±")
        
        p.append(float(pref))
        ep.append(float(epref))
        
        t.append(float(thetaref))
        et.append(float(ethetaref))

    wl, ewl = np.array(wl),np.array(ewl)
    p, ep = np.array(p),np.array(ep)
    t, et = np.array(t),np.array(et)
    
    ref_pol["WL"], ref_pol["EWL"] = wl, ewl
    ref_pol["POL"], ref_pol["EPOL"] = p, ep
    ref_pol["THETA"], ref_pol["ETHETA"] = t, et

    ref_pol["BANDS"] = s_bands
    
    return ref_pol


def plot_s4data_for_polarimetry(objprods, object_indexes, percentile=98, aperture_radius=10, object_name="") :


    """ Module to plot differential light curves and auxiliary data
    
    Parameters
    ----------
    objprods: dict
        data container to store products of analysis
    object_indexes: list: [int, int, int, int]
        list of target indexes
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    object_name : str
        object name
    
    Returns
        
    -------
    """

    lcs = []
    nstars = 1

    bands = ['g', 'r', 'i', 'z']
    colors = ['lightblue','darkgreen','darkorange','red']
    panel_pos = [[0,0],[0,1],[1,0],[1,1]]
    
    
    catalog_n_ext = "CATALOG_POL_N_AP{:03d}".format(aperture_radius)
    catalog_s_ext = "CATALOG_POL_S_AP{:03d}".format(aperture_radius)
    
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, layout='constrained', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(8, 8))
        
    for ch in range(4) :
        col, row = panel_pos[ch]
        
        axs[col,row].set_title(r"{}-band".format(bands[ch]))

        if objprods["stack"][ch] is not None :
            
            hdul = fits.open(objprods["stack"][ch])
            img_data, hdr = hdul[0].data, hdul[0].header
            catalog = Table(hdul[catalog_n_ext].data)
            wcs = WCS(hdr, naxis=2)
                
            x_o, y_o = hdul[catalog_n_ext].data['x'], hdul[catalog_n_ext].data['y']
            x_e, y_e = hdul[catalog_s_ext].data['x'], hdul[catalog_s_ext].data['y']

            mean_aper = np.mean(hdul[catalog_n_ext].data['APER'])

            #plt.figure(figsize=(10, 10))

            axs[col,row].plot(x_o, y_o, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
            axs[col,row].plot(x_e, y_e, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)

            for i in range(len(x_o)):
                x = [x_o[i], x_e[i]]
                y = [y_o[i], y_e[i]]
                axs[col,row].plot(x, y, 'w-o', alpha=0.5)
                if i == object_indexes[ch] :
                    axs[col,row].annotate("{}".format(i), [np.mean(x)-25, np.mean(y)+25], color='gray')
                    axs[col,row].annotate('{}'.format(object_name), [np.mean(x)-25, np.mean(y)+25], color='w',fontsize=12, alpha=0.5)
                else :
                    axs[col,row].annotate("{}".format(i), [np.mean(x)-25, np.mean(y)+25], color='w')
        else :
            img_data = np.empty([1024, 1024])
  
        axs[col,row].imshow(img_data, vmin=np.percentile(img_data, 100-percentile),vmax=np.percentile(img_data, percentile), origin='lower', aspect='equal')
                  
        #axs[col,row].imshow(img_data, vmin=np.percentile(img_data, 100. - percentile), vmax=np.percentile(img_data, percentile), origin='lower', cmap='cividis', aspect='equal')
                
        axs[col,row].tick_params(axis='x', labelsize=10)
        axs[col,row].tick_params(axis='y', labelsize=10)
        axs[col,row].minorticks_on()
        axs[col,row].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs[col,row].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
                         
    plt.show()
        
    return


def match_catalog_with_simbad(obj_id, s4_proc_frame, catalog_ext="CATALOG_POL_N_AP010", tolerance=2., verbose=False) :

    """ Module to match a catalog from sci image product (proc.fits) with Simbad
    Parameters
    ----------
    obj_id : str
        object identification name
    s4_proc_frame : str
        sparc4 image fits product (proc.fits) file path
    catalog_ext : str or int (optional)
        FITS hdu extension name or number for catalog
    tolerance : float (optional)
        maximum tolerance for angular distance to match object [arcsec]
    verbose : bool (optional)
        to print messages
    Returns
        imin, coord : tuple : int, SkyCoord
        imin is the object matched index in the catalog and coord is its respective coodinates
    -------
    """

    # query SIMBAD repository to match object by name
    obj_match_simbad = Simbad.query_object(obj_id)
        
    # cast coordinates into SkyCoord
    coord = SkyCoord(obj_match_simbad["RA"][0], obj_match_simbad["DEC"][0], unit=(u.hourangle, u.deg), frame='icrs')
    
    # open sparc4 image product as hdulist
    hdul = fits.open(s4_proc_frame)
    
    # read catalog data as Table
    catalog = Table(hdul[catalog_ext].data)

    # search catalog objects with minimum distance from simbad object
    delta_min, imin  = tolerance, -1
    for i in range(len(catalog)) :
        cat_ra, cat_dec = catalog['RA'][i], catalog['DEC'][i]
        
        dra = np.cos(coord.dec.deg*np.pi/180.)*(cat_ra - coord.ra.deg)*60*60
        ddec = (cat_dec - coord.dec.deg)*60*60
        delta = np.sqrt((dra)**2 + (ddec)**2)
        
        if delta < delta_min :
            delta_min = delta
            imin = i
   
    if imin == -1 :
        print("Object {} did not match a Simbad object".format(obj_id))
        return None, None
   
    if verbose :
        ra_diff = (catalog[imin][1]-coord.ra.deg)*60*60
        dec_diff = (catalog[imin][2]-coord.dec.deg)*60*60

        print("Object {} index is {} with RA={:.5f} (diff={:.2f} arcsec) / DEC={:.5f} (diff={:.2f} arcsec)".format(obj_id, imin, catalog[imin][1], ra_diff, catalog[imin][2], dec_diff))

    return imin, coord
    
    
def match_object_with_simbad(obj_id, ra=None, dec=None, search_radius_arcsec=10) :

    """ Module to match a given object id with Simbad
    Parameters
    ----------
    objprods: dict
        data container to store products
    ra: float (optional)
        right ascension (deg)
    dec: float (optional)
        declination (deg)
    search_radius_arcsec: float
        search radius in units of arcseconds
    Returns
        obj_match_simbad, coord: simbad_entry, SkyCoord()
    -------
    """

    obj_match_simbad, coord = None, None
    
    try :
        print("Querying SIMBAD database to match object ID={}".format(obj_id))
        # query SIMBAD repository to match object by ID
        obj_match_simbad = Simbad.query_object(obj_id)
    except :
        if ra is not None and dec is not None :
            print("Querying SIMBAD database to match an object at RA={} DEC={}".format(ra, dec))
            # cast input coordinates into SkyCoord
            coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
            # query SIMBAD repository to match an object by coordinates
            obj_match_simbad = Simbad.query_region(coord, radius = search_radius_arcsec * (1./3600.) * u.deg)
        else :
            print("WARNING: could not find Simbad match for object {}".format(obj_id))

    if obj_match_simbad is not None :
        ra = obj_match_simbad["RA"][0]
        dec = obj_match_simbad["DEC"][0]
    
        # cast coordinates into SkyCoord
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')

    return obj_match_simbad, coord

    
def get_object_indexes_in_catalogs(objprods, catalog_ext=1, simbad_id=None, verbose=False) :

    """ Module to get object indexes in catalogs of s4 image products
    Parameters
    ----------
    objprods: dict
        data container to store products
    catalog_ext: int or str
        FITS extension for catalog
    simbad_id: str
        simbad identification name
    verbose: bool
        print verbose messages
    Returns
        object_indexes : list : [int, int, int, int]
        list of detected object indexes in the four channels
    -------
    """

    object_indexes = [None, None, None, None]
    
    if simbad_id is None :
        simbad_id = objprods["object"]
        
    for ch in range(4) :
        if objprods["stack"][ch] :
            imin, coord = match_catalog_with_simbad(simbad_id, objprods["stack"][ch], catalog_ext=catalog_ext, verbose=verbose)
            if imin is not None :
                object_indexes[ch] = imin
                
    return object_indexes



parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help='Input data directory',type='string', default="./")
parser.add_option("-j", "--object", dest="object", help='Object ID',type='string', default="")
parser.add_option("-x", "--obj_indexes", dest="obj_indexes", help='Object indexes',type='string', default="")
parser.add_option("-o", "--output", dest="output", help='Output master catalog',type='string', default="")
parser.add_option("-a", "--aperture", dest="aperture", help='Aperture radius for polarimetry',type='string', default="")
parser.add_option("-r", "--ref_pol", dest="ref_pol", help='Reference polarization values',type='string', default="")
parser.add_option("-t", "--ref_theta", dest="ref_theta", help='Reference polarization angle values',type='string', default="")
parser.add_option("-b", "--ref_bands", dest="ref_bands", help='Reference bands',type='string', default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with sparc4_polarimetry_standard.py -h "); sys.exit(1);

calibrate_photometry = False
calibrate_astrometry = False
create_master_catalog = False
plot_polar_map = True
get_L2_polar = True
get_L4_polar = False

# input data directory
datadir = options.input

# load SPARC4 data products in the directory
s4products = load_data_products(datadir, object_id=options.object)

objprods={}
for key in s4products.keys() :
    if 'POLAR' in key :
        objprods = s4products[key]
        print("Mode (key) : {}".format(key))
        for ch in range(4) :
            print("******* CHANNEL {} ***********".format(ch+1))
            print(objprods["object"], "STACK: ", objprods["stack"][ch])
            print(objprods["object"], "POLAR L2: ", objprods["polarl2"][ch])
            print(objprods["object"], "POLAR L4: ", objprods["polarl4"][ch])
            print(objprods["object"], "TS: ", objprods["ts"][ch])
            print(objprods["object"], "LC: ", objprods["lc"][ch])
            print("\n")
        # break to get only the first set of files
        break

aperture_radius=12

# find Simbad match for main object
object_match_simbad, object_coords = match_object_with_simbad(options.object, search_radius_arcsec=10)
    
object_indexes = []
if options.obj_indexes == "" :
    #  identify main source in the observed catalogs
    object_indexes = get_object_indexes_in_catalogs(objprods, obj_id=options.object, simbad_id=options.object)
else :
    objidx = options.obj_indexes.split(",")
    for i in range(len(objidx)) :
        object_indexes.append(int(objidx[i]))
print(object_indexes)
    
if options.plot :
    plot_s4data_for_polarimetry(objprods, object_indexes, object_name=options.object)

catalog_ext = "CATALOG_POL_N_AP{:03d}".format(aperture_radius)

bands = ['g','r','i','z']

p, p_err = np.array([]), np.array([])
theta, theta_err = np.array([]), np.array([])
wl, wl_err = np.array([]), np.array([])

aperture_radius = None
if options.aperture != "":
    aperture_radius = float(options.aperture)

for ch in range(4) :
    if objprods["polarl2"][ch] is not None and get_L2_polar :
        print(objprods["polarl2"][ch])
        pol_results = s4pipelib.get_polarimetry_results(objprods["polarl2"][ch], source_index=object_indexes[ch], aperture_radius=aperture_radius, min_aperture=0, max_aperture=21, compute_k=True, plot=options.plot, verbose=options.verbose)
        qlab = r"{:.4f}$\pm${:.4f}".format(100*pol_results["Q"].nominal, 100*pol_results["Q"].std_dev)
        ulab = r"{:.4f}$\pm${:.4f}".format(100*pol_results["U"].nominal, 100*pol_results["U"].std_dev)
        plab = r"{:.4f}$\pm${:.4f}".format(100*pol_results["P"].nominal, 100*pol_results["P"].std_dev)
        thetalab = r"{:.2f}$\pm${:.2f}".format(pol_results["THETA"].nominal, pol_results["THETA"].std_dev)
        print("{} & {} & {} & {} & {}".format(bands[ch],qlab,ulab,plab,thetalab))
        
        p = np.append(p,pol_results["P"].nominal)
        p_err = np.append(p_err,pol_results["P"].std_dev)
        
        theta = np.append(theta,pol_results["THETA"].nominal)
        theta_err = np.append(theta_err,pol_results["THETA"].std_dev)

        wl_eff = get_bandpasses()[bands[ch]]

        wl = np.append(wl,wl_eff.nominal_value)
        wl_err = np.append(wl_err,wl_eff.std_dev)

ref_pol = get_ref_data(options.ref_pol, options.ref_theta, options.ref_bands)
wlref, wlref_err = ref_pol["WL"], ref_pol["EWL"]
thetaref, thetaref_err = ref_pol["THETA"], ref_pol["ETHETA"]
pref, pref_err = ref_pol["POL"], ref_pol["EPOL"]

run_mcmc = True

# First Fit Serkowski law to polarimetry as function of wavelength
pfit, errors, samples = fit_Serkowski(ref_pol, run_mcmc=run_mcmc)
wl_max, p_max, k = pfit[0], pfit[1], pfit[2]

wl_model = np.linspace(300, 1000, 100)
p_model = serkowski(wl_max, p_max, k, wl_model)

# Start plotting Serkowski fit.
fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [2, 1]})
        
#plt.title("{}".format(options.object),fontsize=20)
axs[0].set_title(r"{}".format(options.object), fontsize=20)

if run_mcmc :
    nsamples = 300
    polsamplemodels = []
    for coeffs in samples[np.random.randint(len(samples), size=nsamples)]:
        polm = serkowski(coeffs[0], coeffs[1], coeffs[2], wl_model)
        polsamplemodels.append(polm)
    polsamplemodels = np.array(polsamplemodels, dtype=float)
    rms = np.nanstd(polsamplemodels,axis=0)
    axs[0].fill_between(wl_model, p_model+rms, p_model-rms, color="darkgreen", alpha=0.3, edgecolor="none")


refbands = options.ref_bands.split(",")
uniquerefbands = []
refbands_str = ""
for passband in refbands :
    if passband not in uniquerefbands :
        if len(uniquerefbands) == 0 :
            refbands_str += "{}".format(passband)
        else :
            refbands_str += ",{}".format(passband)
        uniquerefbands.append(passband)


axs[0].plot(wl_model, p_model, "-", lw=1.5, color="darkgreen", label=r"Serkowski:$\lambda_m=${:.1f} nm; p$_m=${:.1f}%; k={:.1f}".format(wl_max, p_max, k))
axs[0].errorbar(wlref,pref,xerr=wlref_err,yerr=pref_err,fmt="o",label="Reference ({})".format(refbands_str))
axs[0].errorbar(wl,p*100.,xerr=wl_err,yerr=p_err*100.,fmt="o",label="SPARC4 (g,r,i,z)")

axs[0].tick_params(axis='x', labelsize=16)
axs[0].tick_params(axis='y', labelsize=16)
axs[0].minorticks_on()
axs[0].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
axs[0].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)

#plt.tick_params(axis='x', labelsize=16)
#plt.tick_params(axis='y', labelsize=16)
#plt.minorticks_on()
#plt.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
#plt.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)

#plt.ylabel(r"Polarization (%)", fontsize=18)
#plt.xlabel(r"$\lambda$ (nm)", fontsize=18)
#plt.legend(fontsize=14)

axs[0].set_ylabel(r"Polarization (%)", fontsize=18)
axs[0].legend(fontsize=14)

pobs_model = serkowski(wl_max, p_max, k, wl)
pobs_ref_model = serkowski(wl_max, p_max, k, wlref)

if run_mcmc :
    axs[1].fill_between(wl_model, +rms, -rms, color="darkgreen", alpha=0.3, edgecolor="none")

axs[1].errorbar(wlref,pref-pobs_ref_model,xerr=wlref_err,yerr=pref_err,fmt="o")
axs[1].errorbar(wl,p*100.-pobs_model,xerr=wl_err,yerr=p_err*100.,fmt="o")

axs[1].tick_params(axis='x', labelsize=16)
axs[1].tick_params(axis='y', labelsize=16)
axs[1].minorticks_on()
axs[1].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
axs[1].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)

axs[1].set_ylabel(r"resid. (%)", fontsize=18)
axs[1].set_xlabel(r"$\lambda$ (nm)", fontsize=18)

plt.show()


# Now calculate delta theta
print("------------------------------------------------------------------------")
theta_ref_mean = ufloat(0, 0)
for i in range(len(thetaref)) :
    utheta_ref = ufloat(thetaref[i], thetaref_err[i])
    theta_ref_mean += utheta_ref/len(thetaref)
print("\tMean theta ref: {} deg\n".format(theta_ref_mean))

print("Angle of polarization correction = (theta_obs - mean_theta_ref) :")
print("------------------------------------------------------------------------")

for i in range(len(thetaref)):
    utheta_ref = ufloat(thetaref[i], thetaref_err[i])
    print("\tBand {}: {} deg".format(refbands[i], utheta_ref-theta_ref_mean))
print("------------------------------------------------------------------------")
for i in range(4) :
    utheta = ufloat(theta[i],theta_err[i])
    print("\tBand {}: {} deg".format(bands[i], utheta-theta_ref_mean))
print("------------------------------------------------------------------------")

    
