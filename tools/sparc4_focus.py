"""
    Created on Jan 23 2024

    Description: Module to run focus of SPARC4

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python -W"ignore" /Users/eder/sparc4-pipeline/tools/sparc4_focus.py --datadir="/Users/eder/Data/SPARC4/focus_data/" --reducedir="/Users/eder/Data/SPARC4/focus_data/reduced" --nightdir=20231101 --seq_suffix=focusseq -p
    
    """

import os, sys
from optparse import OptionParser

import numpy as np
import astropy.io.fits as fits

import sparc4.db as s4db
import sparc4.pipeline_lib as s4pipelib

#from astropop.photometry import background, starfind
import glob

from photutils.detection import DAOStarFinder

import photutils
from astropy.stats import SigmaClip

from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

from astropy.modeling import models, fitting

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def measure_psf(filename, threshold=3., focus_value_key="TELFOCUS", read_noise_key="RDNOISE",  aperture_radius=50, fwhm_for_source_detection=5.0, convolve_data=True, use_moffat=False, plot=False, verbose=False) :

    #open fits image
    hdul = fits.open(filename)
    
    ext = 0
    if filename.endswith(".fits.fz") :
        ext = 1
    
    hdr = hdul[ext].header
    img_data = hdul[ext].data
    err_data = np.sqrt(img_data + hdr[read_noise_key]*hdr[read_noise_key])

    # get focus position value from header
    focus_position = float(hdr[focus_value_key])
    
    # calculate background
    #bkg, rms = background(img_data, global_bkg=True)
    
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(img_data, (50, 50), filter_size=(5, 5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    sig = bkg.background_rms_median
    rms = bkg.background_rms
    
    # subtract background
    img_data = np.array(img_data, dtype=float) - bkg.background
    # update error data
    err_data = np.sqrt(err_data*err_data + rms*rms)
    
    convolved_data = img_data
    # Convolve data with a 2D Gaussian
    if convolve_data :
        kernel = make_2dgaussian_kernel(fwhm_for_source_detection, size=5)
        convolved_data = convolve(img_data, kernel)
        err_data = convolve(err_data, kernel)

    ######### SEGMENTATION MAP ##########
    # detect sources using a segmentation map
    #finder = SourceFinder(npixels=5, progress_bar=False)
    #segment_map = finder(convolved_data, threshold)
    # create a catalog of sources out of the segmentation matp
    #cat = SourceCatalog(img_data, segment_map, convolved_data=convolved_data)
    #nsources = len(cat)
    # set array of tuples with x,y positions of sources
    #positions = [( ((cat.bbox_xmin+cat.bbox_xmax)/2)[i],((cat.bbox_ymin+cat.bbox_ymax)/2)[i]) for i in range(len(cat.bbox_xmin))]
    ####################################
    
    ######### DAOFind ##########
    # initialize daofinder
    daofind = DAOStarFinder(threshold=threshold*sig, fwhm=fwhm_for_source_detection, min_separation=fwhm_for_source_detection)
    
    # detect sources
    sources = daofind.find_stars(convolved_data)
    # set number of sources from daofind
    nsources = len(sources)
    if verbose :
        print("DAOFind threshold: {} e-  FHWM: {} pix; detected {} sources".format(threshold*sig, fwhm_for_source_detection, nsources))
    # set array of tuples with x,y positions of sources
    positions = [(sources['xcentroid'][i],sources['ycentroid'][i]) for i in range(len(sources['xcentroid']))]
    ####################################
        
    # set circular apertures for photometry
    apertures = photutils.aperture.CircularAperture(positions, r=aperture_radius)
    # calculate photometric quantities for all sources
    aper_stats = photutils.aperture.ApertureStats(img_data, apertures, error=err_data)

    fwhm, fwhmx, fwhmy = np.array([]),np.array([]),np.array([])
    sigma_to_fwhm = 2*np.sqrt(2*np.log(2))
        
    for i in range(1,nsources) :
    
        x1,x2 = aper_stats.bbox_xmin[i],aper_stats.bbox_xmax[i]
        y1,y2 = aper_stats.bbox_ymin[i],aper_stats.bbox_ymax[i]
           
        if verbose :
            print("idx={} / {} x={} y={}".format(i,nsources,sources['xcentroid'][i],sources['ycentroid'][i]))
        box_data = img_data[y1:y2,x1:x2]
        
        xvalues, yvalues = np.mean(box_data,0), np.mean(box_data,1)
        xcoords, ycoords = np.arange(len(xvalues)), np.arange(len(yvalues))
                
        if len (xcoords) and len (ycoords):
    
            xvalues -= np.min(xvalues)
            yvalues -= np.min(yvalues)

            max_value = np.nanmax(box_data)

            fit_g = fitting.LevMarLSQFitter()
            if use_moffat :
                g_init = models.Moffat1D(amplitude=max_value, x_0=np.nanargmax(xvalues), gamma=fwhm_for_source_detection, alpha=1)
            else :
                g_init = models.Gaussian1D(amplitude=max_value, mean=np.nanargmax(xvalues), stddev=fwhm_for_source_detection)
            gx = fit_g(g_init, xcoords, xvalues, maxiter=200)
         
            if use_moffat :
                x_fwhm = gx.fwhm
            else :
                x_fwhm = sigma_to_fwhm * gx.stddev.value

            fit_g = fitting.LevMarLSQFitter()
            if use_moffat :
                g_init = models.Moffat1D(amplitude=max_value, x_0=np.nanargmax(yvalues), gamma=fwhm_for_source_detection, alpha=1)
            else :
                g_init = models.Gaussian1D(amplitude=max_value, mean=np.nanargmax(yvalues), stddev=fwhm_for_source_detection)
            gy = fit_g(g_init, ycoords, yvalues, maxiter=200)

            if use_moffat :
                y_fwhm = gy.fwhm
            else :
                y_fwhm = sigma_to_fwhm * gy.stddev.value
                
            fwhmx = np.append(fwhmx,x_fwhm)
            fwhmy = np.append(fwhmy,y_fwhm)
            fwhm = np.append(fwhm, np.nanmean(np.array([x_fwhm,y_fwhm])))

            if plot :
                plt.plot(xcoords,xvalues,"g.")
                plt.plot(xcoords,gx(xcoords),"g-",lw=2,label="Fit in x-direction")
            
                plt.plot(ycoords,yvalues,"r.")
                plt.plot(ycoords,gy(ycoords),"r-",lw=2,label="Fit in y-direction")
            
                #plt.imshow(box_data)
                plt.show()
                
    if plot :
        plt.plot(fwhm,"o")
        plt.show()

    return focus_position, fwhm
    

def hyperbolic_function(x, a, b, c, d):
    return d + a * np.sqrt((((x - c)**2)/b**2) + 1)

def hyperbolic_fit(xdata, ydata):
    # Initial guess for the parameters
    initial_guess = [1, 1, np.nanmedian(xdata), 1]
    # Performing the curve fitting
    params, params_covariance = curve_fit(hyperbolic_function, xdata, ydata, p0=initial_guess)
    return params


parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",help="SPARC4 channels: e.g '1,3,4' ", type='string',default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",help="Name of night directory common to all channels",type='string', default="")
parser.add_option("-s", "--seq_suffix", dest="seq_suffix",help="Suffix used to select focus images",type='string', default="")
parser.add_option("-m", "--psf_function", dest="psf_function",help="PSF function: Gaussian or Moffat",type='string', default="Moffat")
parser.add_option("-w", "--window", dest="window",help="Window aperture size in units of FWHM",type='float', default=5.)
parser.add_option("-f", "--fwhm", dest="fwhm",help="Estimated FWHM (pixels)",type='float', default=5.)
parser.add_option("-k", "--focus_keyword", dest="focus_keyword",help="Keyword with focus values",type='string', default="TELFOCUS")
parser.add_option("-t", "--threshold", dest="threshold",help="Threshold (sigmas) to detect sources on focus images",type='float', default=10.)
parser.add_option("-e", "--platescale", dest="platescale",help="Plate scale in arcsec/pixel",type='float', default=0.335)
parser.add_option("-l", action="store_true", dest="parabolic_fit",help="to use parabolic fit", default=False)
parser.add_option("-p", action="store_true", dest="plot",help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_focus.py")
    sys.exit(1)

use_moffat = False
if options.psf_function == "Moffat" :
    use_moffat = True
elif options.psf_function == "Gaussian" :
    use_moffat = False
else :
    print("WARNING: unsupported PSF function: {}. Using Gaussian.".format(options.psf_function))
    
# potential keyword to select focus data automatically -- may not be useful as people set "FOCUS" mode for testing
obstype_focus_key = "FOCUS"

# initialize pipeline parameters
p = s4pipelib.init_s4_p(options.nightdir,
                        options.datadir,
                        options.reducedir,
                        options.channels,
                        print_report=options.verbose)

channel_colors = p['CHANNEL_COLORS']
channel_labels = p['CHANNEL_LABELS']
nchannels = len(p['SELECTED_CHANNELS'])

ncols, nrows = 2, 2
panelpos = [[0,0],[0,1],[1,0],[1,1]]
if nchannels == 1 :
    ncols, nrows = 1, 1
elif nchannels == 2 :
    ncols, nrows = 2, 1
    panelpos = [0,1]
if nchannels == 3 :
    ncols, nrows = 1, 3
    panelpos = [0,1,2]

if options.plot :
    # plot best polarimetry results
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), sharex=False, sharey=False, gridspec_kw={
                             'hspace': 0.5})

idx = 0

best_focus_values = np.array([])
best_fwhm = np.array([])
best_channels = []

# Run full reduction for selected channels
for channel in p['SELECTED_CHANNELS']:

    # set zero based index of current channel
    j = channel - 1

    print("******************************")
    print("Measuring FOCUS for CHANNEL {}".format(channel))
    print("******************************")

    data_dir = p['data_directories'][j]
    reduce_dir = p['reduce_directories'][j]

    pattern = "{}/*_{}.fits".format(data_dir,options.seq_suffix)
    
    inputdata = sorted(glob.glob(pattern))
    
    if len(inputdata) == 0 :
        print("WARNING: Could not identify any image in the path: {}".format(data_dir))
        continue
    
    focus_values = np.array([])
    fwhms, fwhms_err = np.array([]), np.array([])
    
    for i in range(len(inputdata)) :
        basename = os.path.basename(inputdata[i])
        print("processing image {} / {} : {}".format(i+1,len(inputdata),basename))

        try :
            focus_position, fwhm = measure_psf(inputdata[i], threshold=options.threshold, focus_value_key=options.focus_keyword,  aperture_radius=options.window*options.fwhm,fwhm_for_source_detection=options.fwhm, use_moffat=use_moffat)
        except :
            print("Could not measure focus on image {}, skipping ...".format(inputdata[i]))
            continue
            
        if len(fwhm) :
            mfwhm = np.nanmedian(fwhm)
            fwhm_err = np.nanmedian(np.abs(fwhm-mfwhm)) / 0.67449
                 
            fwhms = np.append(fwhms,mfwhm)
            fwhms_err = np.append(fwhms_err,fwhm_err)
            focus_values = np.append(focus_values,focus_position)

            print("Image {} ({}/{}): detected {} sources; focus position={:.1f}; FWHM={:.2f}+/-{:.2f} pixels".format(basename,i+1,len(inputdata),len(fwhm),focus_position, mfwhm, fwhm_err))
        else :
            print("WARNING: failed to detect sources on focus image: {}".format(basename))
    
    if options.plot :
        if nchannels == 1:
            ax = axes
        elif nchannels == 2 or nchannels == 3:
            ax = axes[panelpos[idx]]
        else :
            ax = axes[panelpos[idx][0],panelpos[idx][1]]
        ax.set_title("FOCUS model for CHANNEL {} ({}-band)".format(channel,channel_labels[j]))
        ax.errorbar(focus_values,fwhms,yerr=fwhms_err,fmt="ko", alpha=0.1)
        ax.plot(focus_values,fwhms,"ko",label="Data")
    
    xs = np.linspace(np.nanmin(focus_values), np.nanmax(focus_values), 300)
    ymin, ymax = np.nanmin(fwhms)-1,np.nanmax(fwhms)+1
    
    if options.parabolic_fit :
        # fit parabola
        coeffs = np.polyfit(focus_values, fwhms, 2)
        parabola = np.poly1d(coeffs)
    
        dydx = np.gradient(parabola(xs))
        imin = np.argmin(np.abs(dydx))
    
        if options.plot :
            #ax.plot(xs,parabola(xs), "-", color=channel_colors[j], lw=1, label="f(x) = {:.4f} + {:.4f}*x".format(coeffs[1],coeffs[0]))
            ax.plot(xs,parabola(xs), "-", color=channel_colors[j], lw=1, label="parabolic fit")
            ax.vlines(xs[imin], ymin, ymax, color="red", ls=":", label="Min focus model at {:.2f}".format(xs[imin]))

        best_focus_values = np.append(best_focus_values, xs[imin])
        best_channels.append(channel)
        best_fwhm = np.append(best_fwhm, parabola(xs)[imin])
    else :
        # fit hyperbole
        coeffs_hyp = hyperbolic_fit(focus_values, fwhms)
        hyperbolic = hyperbolic_function(focus_values, *coeffs_hyp)
        
        dydx_h = np.gradient(hyperbolic_function(xs, *coeffs_hyp))
        imin_h = np.argmin(np.abs(dydx_h))

        if options.plot :
            #ax.plot(xs,hyperbolic_function(xs, *coeffs_hyp), "-", color=channel_colors[j], lw=2, label="f(x) = {:.4f} + {:.4f}*sqrt((x - {:.4f})**2)/{:.4f}**2 + 1)".format(coeffs_hyp[3],coeffs_hyp[0],coeffs_hyp[2],coeffs_hyp[3]))
            ax.plot(xs,hyperbolic_function(xs, *coeffs_hyp), "-", color=channel_colors[j], lw=2, label="hyperbolic fit")
            ax.vlines(xs[imin_h], ymin, ymax, color="red", ls=":", label="Min focus model at {:.2f}".format(xs[imin_h]))
        
        best_focus_values = np.append(best_focus_values, xs[imin_h])
        best_channels.append(channel)
        best_fwhm = np.append(best_fwhm, hyperbolic_function(xs, *coeffs_hyp)[imin_h])

    minvalue = np.nanargmin(fwhms)
    
    if options.plot :
        ax.plot([focus_values[minvalue]], [fwhms[minvalue]], "rx", lw=2, label="Min focus value at {:.2f}".format(focus_values[minvalue]))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.minorticks_on()
        ax.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        ax.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
        ax.set_xlabel("Focus value (a.u.)",fontsize=18)
        ax.set_ylabel("FWHM (pixel)",fontsize=18)
        ax.legend()
    
    idx += 1
    
print("\n-------------------")
print("FINAL FOCUS RESULTS")
print("-------------------")
for i in range(len(best_channels)) :
    print("CHANNEL {}: best focus at {:.2f} -> FWHM={:.2f} pixels or {:.2f} arcsec".format(best_channels[i],best_focus_values[i],best_fwhm[i],best_fwhm[i]*options.platescale))

best_mean_fwhm = np.mean(best_fwhm)
print("***************************")
print(" Best mean FOCUS at {:.2f} with FWHM={:.2f} pixels or {:.2f} arcsec".format(np.mean(best_focus_values),best_mean_fwhm,best_mean_fwhm*options.platescale))
print("***************************")

if idx and options.plot :
    plt.show()
