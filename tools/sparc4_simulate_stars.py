"""
    Created on Nov 14 2024

    Description: Module to simulate stars to test pipeline

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python tools/sparc4_simulate_stars.py --input=/Users/eder/Data/SPARC4/simu_star_test/sparc4acs?/20240618/*_toi1853.fits --output_src_file=/Users/eder/Data/SPARC4/simu_star_test/simul_src.csv --suffix="_mod" --number_of_stars=10 -ov
    
    """

import os, sys
from optparse import OptionParser

import glob
import astropy.io.fits as fits

import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, Moffat2DKernel
import numpy as np
from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image
    
def stars(number, max_counts=10000, gain=1, fwhm=3, y_max=1024, x_max=1024) :
    """
    Add some stars to the image.
    """
    
    flux_range = [max_counts/10, max_counts]

    xmean_range = [0.1 * x_max, 0.9 * x_max]
    ymean_range = [0.1 * y_max, 0.9 * y_max]
    
    sigma_to_fwhm = 2.*np.sqrt(2.*np.log(2.))
        
    # convert fwhm to standard dev
    stddev = fwhm / sigma_to_fwhm
    
    xstddev_range = [stddev, stddev]
    ystddev_range = [stddev, stddev]
    params = dict([('amplitude', flux_range),
                  ('x_mean', xmean_range),
                  ('y_mean', ymean_range),
                  ('x_stddev', xstddev_range),
                  ('y_stddev', ystddev_range),
                  ('theta', [0, 2*np.pi])])

    sources = make_random_gaussians_table(number, params,seed=12345)
    
    return sources
    

parser = OptionParser()
parser.add_option("-i", "--input", dest="input",help="Input pattern to select images to insert simulated stars", type='string', default="*.fits")
parser.add_option("-n", "--number_of_stars", dest="number_of_stars",help="Number of simulated stars", type='int', default=1)
parser.add_option("-s", "--suffix", dest="suffix",help="Suffix to add in modified data", type='string', default="")
parser.add_option("-f", "--fwhm", dest="fwhm",help="FWHM of simulated stars (arcsec)", type='float', default=1.5)
parser.add_option("-t", "--saturation", dest="saturation",help="Saturation limit (ADU)", type='float', default=32000.)
parser.add_option("-p", "--pix_scale", dest="pix_scale",help="Pixel scale (arcsec/pixel)", type='float', default=0.335)
parser.add_option("-r", "--output_src_file", dest="output_src_file",help="Output sources file", type='string', default="")
parser.add_option("-o", action="store_true", dest="overwrite",help="overwrite", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with -h sparc4_simulate_stars.py")
    sys.exit(1)

inputdata = sorted(glob.glob(options.input))

saturation = options.saturation
number_of_stars = options.number_of_stars
pix_scale = options.pix_scale # arcsec / pix
fwhm_arcsec = options.fwhm
fwhm_pix = fwhm_arcsec / pix_scale

if options.verbose :
    print("Simulated FWHM = {} pixels or {} arcsec".format(fwhm_pix,fwhm_arcsec))

base_header = fits.getheader(inputdata[0])
gain = base_header['GAIN']
x_max, y_max = base_header['NAXIS1'], base_header['NAXIS2']

sources = stars(number_of_stars, max_counts=saturation, gain=gain, fwhm=fwhm_pix, y_max=y_max, x_max=x_max)
if options.verbose :
    print(sources)
    
if options.output_src_file != "" :
    sources.write(options.output_src_file, overwrite=True)

#gamma, alpha = 3, 2
#moffat_2D_kernel = Moffat2DKernel(gamma, alpha)

for i in range(len(inputdata)) :
    try :
        outfile = inputdata[i].replace(".fits","{}.fits".format(options.suffix))

        hdul = fits.open(inputdata[i])
        hdr = hdul[0].header
        img_data = hdul[0].data
                
        stars_img_data = make_gaussian_sources_image(img_data.shape, sources)
        
        hdul[0].data = hdul[0].data + stars_img_data
        
        if options.verbose :
            print("{}/{} : src_file={} mod_file={}".format(i+1,len(inputdata),inputdata[i],outfile))
            
        hdul.writeto(outfile, overwrite=options.overwrite)
    
    except Exception as e:
        print("ERROR: could not create image with simulated stars in frame: {}".format(inputdata[i]))
        continue
        
