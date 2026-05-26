"""
    Created on May 23 2026

    Description: Module to monitor raw photometry of SPARC4

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python -W"ignore" /Users/eder/sparc4-pipeline/tools/sparc4_raw_photometry_monitor.py --datadir="/Users/eder/Data/SPARC4/minidata_ql" --nightdir=today --seq_suffix=cr3 -p
    
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

from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u


def measure_raw_photometry(filename, threshold=3., read_noise_key="RDNOISE", time_key="DATE-OBS",  window_size=25, aperture_radius=15, fwhm_for_source_detection=5.0, psf_global_fit=True, plot=False, verbose=False, longitude=-45.5825, latitude=-22.53444, altitude=1864) :

    loc = {}

    #open fits image
    hdul = fits.open(filename)
    
    ext = 0
    if filename.endswith(".fits.fz") :
        ext = 1
    
    hdr = hdul[ext].header
    img_data = hdul[ext].data
    
    observatory_location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=altitude*u.m)
    obstime = Time(hdr[time_key], format='isot', scale='utc', location=observatory_location)
    jd = obstime.jd
    loc["DATE-OBS"] = hdr[time_key]
    loc["JD"] = jd
    
    # calculate background
    #bkg, rms = background(img_data, global_bkg=True)
    
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(img_data, (50, 50), filter_size=(5, 5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    sig = bkg.background_rms_median
    rms = bkg.background_rms
    
    # cast numpy float array
    img_data = np.array(img_data, dtype=float)
    
    # subtract background
    img_minus_bkg_data = img_data - bkg.background
    
    # calculate error data array
    err_data = np.sqrt(img_data)
    if read_noise_key in hdr.keys() :
        err_data = np.sqrt(img_data + hdr[read_noise_key]*hdr[read_noise_key] + rms*rms)
    else :
        err_data = np.sqrt(img_data + rms*rms)
        
    ######### DAOFind ##########
    # initialize daofinder
    daofind = DAOStarFinder(threshold=threshold*sig, fwhm=fwhm_for_source_detection, min_separation=fwhm_for_source_detection)
    #id: unique object identification number.
    #x_centroid, y_centroid: object centroid.
    #sharpness: object sharpness.
    #roundness1: object roundness based on symmetry.
    #roundness2: object roundness based on marginal Gaussian fits.
    #n_pixels: the total number of pixels in the Gaussian kernel array.
    #peak: the peak pixel value of the object.
    #flux: the object instrumental flux calculated as the sum of data values within the kernel footprint.
    #mag: the object instrumental magnitude calculated as -2.5 * log10(flux).
    #daofind_mag:
    
    # detect sources in image minus background
    sources = daofind.find_stars(img_minus_bkg_data)
    
    # set number of sources from daofind
    nsources = len(sources)
    if verbose :
        print("DAOFind threshold: {} e-  FHWM: {} pix; detected {} sources".format(threshold*sig, fwhm_for_source_detection, nsources))
        
    # set array of tuples with x,y positions of sources
    positions = [(sources['xcentroid'][i],sources['ycentroid'][i]) for i in range(len(sources['xcentroid']))]
    ####################################
        
    # set circular apertures for photometry
    apertures = photutils.aperture.CircularAperture(positions, r=aperture_radius)
    # calculate photometric quantities for all sources in raw image
    raw_aper_stats = photutils.aperture.ApertureStats(img_data, apertures, error=err_data)
    # calculate photometric quantities for all sources
    aper_stats = photutils.aperture.ApertureStats(img_minus_bkg_data, apertures, error=err_data)

    # measure fwhms and updated centroids
    fwhms_x, fwhms_y, xc, yc = s4pipelib.measure_fwhm_from_2DGaussianFit(img_minus_bkg_data, positions, err_data=err_data, window_size=window_size, global_fit=psf_global_fit, plot=False, verbose=False)
    # measure fhwms
    #fwhms = measure_fwhm(img_minus_bkg_data, positions, window_size=aperture_radius, plot=False, verbose=False)
    
    fluxes = aper_stats.sum
    fluxerrs = aper_stats.sum_err
    max_counts = raw_aper_stats.max

    mfwhm_x, mfwhm_y = np.nanmedian(fwhms_x), np.nanmedian(fwhms_y)
    mfwhm = (mfwhm_x+mfwhm_y)/2
    fwhm_x_err = np.nanmedian(np.abs(fwhms_x-mfwhm_x)) / 0.67449
    fwhm_y_err = np.nanmedian(np.abs(fwhms_y-mfwhm_y)) / 0.67449
    
    loc["fwhms_x"] =  fwhms_x
    loc["fwhms_y"] = fwhms_y
    loc["fwhm_x"] = mfwhm_x
    loc["fwhm_y"] = mfwhm_y
    loc["fwhm_x_err"] = fwhm_x_err
    loc["fwhm_y_err"] = fwhm_y_err
    loc["fwhm"] = mfwhm
    loc["fwhm_err"] = np.sqrt(fwhm_x_err*fwhm_x_err+fwhm_y_err*fwhm_y_err)
    loc["fluxes"] = fluxes
    loc["fluxerrs"] = fluxerrs
    loc["max_counts"] = max_counts

    return loc
    

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",help="SPARC4 channels: e.g '1,3,4' ", type='string',default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",help="Name of night directory common to all channels",type='string', default="")
parser.add_option("-s", "--seq_suffix", dest="seq_suffix",help="Suffix used to select images",type='string', default="")
parser.add_option("-w", "--window", dest="window",help="Window size in units of pixels",type='int', default=25)
parser.add_option("-f", "--fwhm", dest="fwhm",help="Estimated FWHM (pixels)",type='float', default=5.)
parser.add_option("-t", "--threshold", dest="threshold",help="Threshold (sigmas) to detect sources on focus images",type='float', default=10.)
parser.add_option("-e", "--platescale", dest="platescale",help="Plate scale in arcsec/pixel",type='float', default=0.335)
parser.add_option("-p", action="store_true", dest="plot",help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_focus.py")
    sys.exit(1)

    
# initialize pipeline parameters
p = s4pipelib.init_s4_p(options.nightdir,
                        options.datadir,
                        options.reducedir,
                        options.channels,
                        print_report=options.verbose)

channel_colors = p['CHANNEL_COLORS']
channel_labels = p['CHANNEL_LABELS']
nchannels = len(p['SELECTED_CHANNELS'])

best_focus_values = np.array([])
best_fwhm = np.array([])
best_channels = []

# Run full reduction for selected channels
for channel in p['SELECTED_CHANNELS']:

    # set zero based index of current channel
    j = channel - 1

    print("******************************")
    print("Measuring RAW PHOTOMETRY for CHANNEL {}".format(channel))
    print("******************************")

    data_dir = p['data_directories'][j]
    reduce_dir = p['reduce_directories'][j]

    pattern = "{}/*_{}.fits".format(data_dir,options.seq_suffix)
    
    inputdata = sorted(glob.glob(pattern))
    
    if len(inputdata) == 0 :
        print("WARNING: Could not identify any image in the path: {}".format(data_dir))
        continue
    
    fwhms, fwhms_err, flux_values = np.array([]), np.array([]), np.array([])
    
    for i in range(len(inputdata)) :
        basename = os.path.basename(inputdata[i])
        print("processing image {} / {} : {}".format(i+1,len(inputdata),basename))
        
        phot = measure_raw_photometry(inputdata[i], threshold=options.threshold,  window_size=options.window,aperture_radius=3*options.fwhm,fwhm_for_source_detection=options.fwhm, psf_global_fit=True, plot=False, verbose=False)
        
        #except :
        #    print("Could not do photometry on image {}, skipping ...".format(inputdata[i]))
        #    continue
            
        if len(phot['max_counts']) :
            fwhms = np.append(fwhms,phot['fwhm'])
            fwhms_err = np.append(fwhms_err,phot['fwhm_err'])
            flux_values = np.append(flux_values,np.nanmax(phot['max_counts']))

            print("Image {} ({}/{}): detected {} sources; peak={:.1f}; FWHM={:.2f}+/-{:.2f} pixels".format(basename,i+1,len(inputdata),len(phot['max_counts']),np.nanmax(phot['max_counts']),phot['fwhm'], phot['fwhm_err']))
        else :
            print("WARNING: failed to detect sources on focus image: {}".format(basename))
    
