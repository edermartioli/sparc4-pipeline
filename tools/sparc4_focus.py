"""
    Created on Jan 23 2024

    Description: Module to run focus of SPARC4

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python -W"ignore" /Users/eder/sparc4-pipeline/tools/sparc4_focus.py --datadir="/Users/eder/Data/SPARC4/focus_data/" --reducedir="/Users/eder/Data/SPARC4/focus_data/reduced" --nightdir=20231101 --seq_suffix=focusseq -v
    
    """

import os, sys
from optparse import OptionParser

import numpy as np
import astropy.io.fits as fits

import sparc4.db as s4db
import sparc4.pipeline_lib as s4pipelib

from astropop.photometry import background, starfind
import glob

from photutils.detection import DAOStarFinder
import photutils
from astropy.stats import SigmaClip

from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog

from astropy.modeling import models, fitting

import matplotlib.pyplot as plt


def measure_focus(filename, threshold=3., focus_value_key="TELFOCUS", read_noise_key="RDNOISE",  aperture_radius=50, fwhm_for_source_detection=5.0) :

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
    bkg, rms = background(img_data, global_bkg=False)
        
    #calculate median sigma
    sig = np.nanmedian(rms)
    
    # subtract background
    img_data = img_data - bkg
    # update error data
    err_data = np.sqrt(err_data*err_data + rms*rms)

    # Convolve data with a 2D Gaussian
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
    daofind = DAOStarFinder(fwhm=fwhm_for_source_detection, threshold=threshold*sig)
    # detect sources
    sources = daofind(convolved_data)
    # set number of sources from daofind
    nsources = len(sources)
    #print("DAOFind detected {} sources".format(nsources))
    # set array of tuples with x,y positions of sources
    positions = [(sources['xcentroid'][i],sources['ycentroid'][i]) for i in range(len(sources['xcentroid']))]
    ####################################
        
    # set circular apertures for photometry
    apertures = photutils.aperture.CircularAperture(positions, r=aperture_radius)
    # calculate photometric quantities for all sources
    aper_stats = photutils.aperture.ApertureStats(img_data, apertures, error=err_data)

    fwhm, fwhmx, fwhmy = np.array([]),np.array([]),np.array([])
    
    for i in range(1,nsources) :
        x1,x2 = aper_stats.bbox_xmin[i],aper_stats.bbox_xmax[i]
        y1,y2 = aper_stats.bbox_ymin[i],aper_stats.bbox_ymax[i]
            
        box_data = img_data[y1:y2,x1:x2]
        
        xvalues = np.mean(box_data,0)
        xcoords = np.arange(len(xvalues))
        
        yvalues = np.mean(box_data,1)
        ycoords = np.arange(len(yvalues))
        
        if len (xcoords) and len (ycoords):
        
            fit_g = fitting.LevMarLSQFitter()
            g_init = models.Gaussian1D(amplitude=np.nanmax(box_data), mean=np.nanargmax(xvalues), stddev=5.)
            gx = fit_g(g_init, xcoords, xvalues, maxiter=200)
         
            x_fwhm = 2*np.sqrt(2*np.log(2)) * gx.stddev.value

            #plt.plot(xcoords,xvalues,"g.")
            #plt.plot(xcoords,gx(xcoords),"g-",lw=2,label="Gaussian fit in x-direction")

            fit_g = fitting.LevMarLSQFitter()
            g_init = models.Gaussian1D(amplitude=np.nanmax(box_data), mean=np.nanargmax(yvalues), stddev=5.)
            gy = fit_g(g_init, ycoords, yvalues, maxiter=200)
         
            #plt.plot(ycoords,yvalues,"r.")
            #plt.plot(ycoords,gy(ycoords),"r-",lw=2,label="Gaussian fit in y-direction")

            y_fwhm = 2*np.sqrt(2*np.log(2)) * gy.stddev.value

            fwhmx = np.append(fwhmx,x_fwhm)
            fwhmy = np.append(fwhmy,y_fwhm)

            fwhm = np.append(fwhm,(x_fwhm + y_fwhm)/2)
        
        #plt.imshow(box_data)
        #plt.show()
        
    #plt.plot(fwhm,"o")
    #plt.show()
    
    return focus_position, fwhm
    

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",help="SPARC4 channels: e.g '1,3,4' ", type='string',default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",help="Name of night directory common to all channels",type='string', default="")
parser.add_option("-s", "--seq_suffix", dest="seq_suffix",help="Suffix used to select focus images",type='string', default="")
parser.add_option("-k", "--focus_keyword", dest="focus_keyword",help="Keyword with focus values",type='string', default="TELFOCUS")
parser.add_option("-t", "--threshold", dest="threshold",help="Threshold (sigmas) to detect sources on focus images",type='float', default=10.)
parser.add_option("-p", action="store_true", dest="plot",help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_focus.py")
    sys.exit(1)

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

# plot best polarimetry results
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), sharex=False, sharey=False, gridspec_kw={
                             'hspace': 0.5})

idx = 0

# Run full reduction for selected channels
for channel in p['SELECTED_CHANNELS']:

    # set zero based index of current channel
    j = channel - 1

    print("******************************")
    print("Measuring FOCUS for CHANNEL {}".format(channel))
    print("******************************")

    data_dir = p['data_directories'][j]
    ch_reduce_dir = p['ch_reduce_directories'][j]
    reduce_dir = p['reduce_directories'][j]

    pattern = "{}/*_{}.fits".format(data_dir,options.seq_suffix)
    
    inputdata = sorted(glob.glob(pattern))
    
    if len(inputdata) == 0 :
        print("WARNING: Could not identify any image in the path: {}".format(data_dir))
        continue
    
    focus_values = np.array([])
    fwhms, fwhms_err = np.array([]), np.array([])
    
    for i in range(len(inputdata)) :

        try :
            focus_position, fwhm = measure_focus(inputdata[i], threshold=options.threshold, focus_value_key=options.focus_keyword)
        except :
            print("Could not measure focus on image {}, skipping ...".format(inputdata[i]))
            continue
            
        if len(fwhm) :
            mfwhm = np.nanmedian(fwhm)
            fwhm_err = np.nanmedian(np.abs(fwhm-mfwhm)) / 0.67449
                 
            fwhms = np.append(fwhms,mfwhm)
            fwhms_err = np.append(fwhms_err,fwhm_err)
            focus_values = np.append(focus_values,focus_position)

            print("Image {} of {} : Focus measured on {} sources at position={:.1f} FWHM = {:.2f} +/- {:.2f} pixels".format(i+1,len(inputdata),len(fwhm),focus_position, mfwhm, fwhm_err))
        else :
            print("WARNING: photometry of focus image {} failed to detect sources, skipping ...".format(inputdata[i]))
    
    # fit parabola
    coeffs = np.polyfit(focus_values, fwhms, 2)
    parabola = np.poly1d(coeffs)
    
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
    ax.plot(xs,parabola(xs), "-", color=channel_colors[j], lw=2, label="f(x) = {:.2f} + {:.2f}*x".format(coeffs[1],coeffs[0]))
    
    minvalue = np.nanargmin(fwhms)
    ax.plot([focus_values[minvalue]], [fwhms[minvalue]], "rx", lw=2, label="Min focus value at {:.2f}".format(focus_values[minvalue]))

    dydx = np.gradient(parabola(xs))
    imin = np.argmin(np.abs(dydx))
    
    ymin, ymax = np.nanmin(fwhms)-1,np.nanmax(fwhms)+1
    ax.vlines(xs[imin], ymin, ymax, color="red", ls=":", label="Min focus model at {:.2f}".format(xs[imin]))
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
    ax.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
    ax.set_xlabel("Focus value (a.u.)",fontsize=18)
    ax.set_ylabel("FWHM (pixel)",fontsize=18)
    ax.legend()
    
    idx += 1

if idx :
    plt.show()
