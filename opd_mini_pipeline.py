"""
    Created on May 6 2023

    Description: Mini pipeline to process OPD data in photometric mode using the SPARC4 pipeline

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python -W"ignore" /Volumes/Samsung_T5/sparc4-pipeline/opd_mini_pipeline.py --bias=/Volumes/Samsung_T5/Data/OPD/WASP-132/raw/*ZERO.fits --flat=/Volumes/Samsung_T5/Data/OPD/WASP-132/raw/*FLAT.fits --science=/Volumes/Samsung_T5/Data/OPD/WASP-132/raw/*WASP-132.fits --reducedir=/Volumes/Samsung_T5/Data/OPD/WASP-132/reduced/ --time_key="DATE" --object="WASP-132" -pv

    python -W"ignore" /Volumes/Samsung_T5/sparc4-pipeline/opd_mini_pipeline.py --bias=/Volumes/Samsung_T5/Data/OPD/WASP-34/fixed/ZERO*.fits --flat=/Volumes/Samsung_T5/Data/OPD/WASP-34/fixed/flat_I*.fits --science=/Volumes/Samsung_T5/Data/OPD/WASP-34fixed/wasp34*.fits --reducedir=/Volumes/Samsung_T5/Data/OPD/WASP-34/reduced/ --time_key="UTDATE" --object="WASP-34" -pv
    
    
    python -W"ignore" /Volumes/Samsung_T5/sparc4-pipeline/opd_mini_pipeline.py --bias=/Volumes/Samsung_T5/Data/OPD/WASP-108/bias*.fits --flat=/Volumes/Samsung_T5/Data/OPD/WASP-108/flat*.fits --science=/Volumes/Samsung_T5/Data/OPD/WASP-108/wasp108b*.fits --reducedir=/Volumes/Samsung_T5/Data/OPD/WASP-108/reduced/ --time_key="DATE-OBS" --object="WASP-108" -vp
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
from optparse import OptionParser

import sparc4_product_plots as s4plt
import sparc4_pipeline_lib as s4pipelib
import sparc4_utils as s4utils
import sparc4_params

import numpy as np

import glob

sparc4_pipeline_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-r", "--reducedir", dest="reducedir", help="Reduced data directory",type='string',default="./")
parser.add_option("-b", "--bias", dest="bias", help="wildcard for bias selection",type='string',default="")
parser.add_option("-F", "--flat", dest="flat", help="wildcard for flat selection",type='string',default="")
parser.add_option("-s", "--science", dest="science", help="wildcard for science data selection",type='string',default="")
parser.add_option("-o", "--object", dest="object", help="object id",type='string',default="unknown_object")
parser.add_option("-t", "--time_key", dest="time_key", help="time keyword",type='string',default="DATE-OBS")
parser.add_option("-a", "--ra", dest="ra", help="RA",type='string',default="")
parser.add_option("-d", "--dec", dest="dec", help="Dec",type='string',default="")
parser.add_option("-f", action="store_true", dest="force", help="Force reduction", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h opd_mini_pipeline.py")
    sys.exit(1)

# load pipeline parameters
p = sparc4_params.load_sparc4_parameters()

# if reduced dir doesn't exist create one
if not os.path.exists(options.reducedir) :
    os.mkdir(options.reducedir)

bias_list = glob.glob(options.bias)
flat_list = glob.glob(options.flat)
sci_list = glob.glob(options.science)

p['master_bias'] = "{}/MasterZero.fits".format(options.reducedir)
p['master_flat'] = "{}/MasterDomeFlat.fits".format(options.reducedir)

# calculate master bias and save product to fits
p = s4pipelib.run_master_calibration(p,
                                     inputlist=bias_list,
                                     output=p['master_bias'],
                                     obstype='bias',
                                     reduce_dir=options.reducedir,
                                     force=options.force)

if options.plot :
    # plot master bias
    s4plt.plot_cal_frame(p["master_bias"], percentile=99.5, combine_rows=True, combine_cols=True)

# calculate master dome flat and save product to FITS
p = s4pipelib.run_master_calibration(p,
                                     inputlist=flat_list,
                                     output=p['master_flat'],
                                     obstype='flat',
                                     reduce_dir=options.reducedir,
                                     normalize=True,
                                     force=options.force)

if options.plot :
    # plot master flat
    s4plt.plot_cal_frame(p["master_flat"], percentile=99.5, xcut=512, ycut=512)

# set reference image for astrometry
p["ASTROM_REF_IMG"] = "{}/calibdb/20230503_s4c3_CR1_astrometryRef_stack.fits".format(sparc4_pipeline_dir)

# set object id
object_id = options.object
# set suffix for stack product
stack_suffix = "{}".format(object_id.replace(" ",""))

if options.time_key != "" :
    p["TIME_KEY"] = options.time_key

# define number of files for stack
p['NFILES_FOR_STACK'] = 10

# define saturation limit
p['SATURATION_LIMIT'] = 65000

# define threshold for source detection
p['PHOT_THRESHOLD'] = 500

# calculate stack
p = s4pipelib.stack_science_images(p,
                                   sci_list,
                                   reduce_dir=options.reducedir,
                                   force=options.force,
                                   stack_suffix=stack_suffix)

if options.plot :
    # plot phot stack product
    s4plt.plot_sci_frame(p['OBJECT_STACK'], nstars=10)

# set number of loops to reduce science data. Each loop must contain a
# given maximum number of frames to avoid memory issues.
nloops = int(np.ceil(len(sci_list) / p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP']))
# set reference image
ref_img = p['REFERENCE_IMAGE']

for j in range(nloops) :

    first = p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] * j
    last = p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] * (j+1)
    if last > len(sci_list) :
        last = len(sci_list)

    print("Running loop {} of {} -> images in loop: {} to {} ... ".format(j,nloops,first,last))

    print("REFERENCE_IMAGE=", ref_img)
    # reduce science data and calculate stack
    p = s4pipelib.reduce_science_images(p,
                                    sci_list[first:last],
                                    reduce_dir=options.reducedir,
                                    ref_img=ref_img,
                                    force=options.force,
                                    match_frames=True,
                                    ra=options.ra,
                                    dec=options.dec)

ts_suffix = "{}".format(object_id.replace(" ",""))

# run photometric time series
phot_ts_product = s4pipelib.phot_time_series(p['OBJECT_REDUCED_IMAGES'][1:],
                                             ts_suffix=ts_suffix,
                                             reduce_dir=options.reducedir,
                                             time_key=p['TIME_KEYWORD_IN_PROC'],
                                             time_format=p['TIME_FORMAT_IN_PROC'],
                                             catalog_names=p['PHOT_CATALOG_NAMES_TO_INCLUDE'],
                                             force=options.force)

target = 0
comps = [1,2,3]

#target = 3
#comps = [1,2,4,5,6,8]

#target = 1
#comps = [0,2,3,4,5,6,8,9]

#if options.plot :
# plot light curve
s4plt.plot_light_curve(phot_ts_product,
                       target=target,
                       comps=comps,
                       nsig=100,
                       plot_coords=True,
                       plot_rawmags=True,
                       plot_sum=True,
                       plot_comps=True,
                       catalog_name=p['PHOT_REF_CATALOG_NAME'])

