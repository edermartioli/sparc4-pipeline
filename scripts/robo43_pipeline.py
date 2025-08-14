"""
    Created on Aug 12 2025


    Description: Mini pipeline to process ROBO43 data using the SPARC4 pipeline

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage examples:

    python /Users/eder/Data/OPD/ROBO40/robo40_fix_kinetic_images.py --input=/Users/eder/Data/ROBO43/20250810/*.fits
    python /Users/eder/Data/OPD/ROBO40/robo40_fix_kinetic_images.py --input=/Users/eder/Data/ROBO43/20250811/*.fits

    python -W"ignore" scripts/robo43_pipeline.py --bias=/Users/eder/Data/ROBO43/20250810/*dark*.fits --flat=/Users/eder/Data/ROBO43/20250811/skyflat_R_*.fits --science=/Users/eder/Data/ROBO43/20250810/M8_R_*.fits --reducedir=/Users/eder/Data/ROBO43/reduced/ --time_key="FRAME" --object="M8" --filter="R" -pv
    """

import glob
import os, sys
from optparse import OptionParser

import sparc4.utils as s4utils
import sparc4.params as s4params
import sparc4.pipeline_lib as s4pipelib
import sparc4.product_plots as s4plt

from astropop.file_collection import FitsFileGroup
import astropy.io.fits as fits
import numpy as np

parser = OptionParser()
parser.add_option("-r", "--reducedir", dest="reducedir", help="Reduced data directory", type='string', default="./")
parser.add_option("-c", "--calibdbdir", dest="calibdbdir", help="Calibration data directory", type='string', default="")
parser.add_option("-R", "--filter", dest="filter", help="Filter name", type='string', default="")
parser.add_option("-b", "--bias", dest="bias", help="wildcard for bias selection", type='string', default="")
parser.add_option("-d", "--dark", dest="dark", help="wildcard for dark selection", type='string', default="")
parser.add_option("-F", "--flat", dest="flat", help="wildcard for flat selection", type='string', default="")
parser.add_option("-s", "--science", dest="science", help="wildcard for science data selection", type='string', default="")
parser.add_option("-o", "--object", dest="object", help="object id", type='string', default="unknown_object")
parser.add_option("-t", "--time_key", dest="time_key", help="time keyword", type='string', default="DATE-OBS")
parser.add_option("-a", "--ra", dest="ra", help="RA", type='string', default="")
parser.add_option("-D", "--dec", dest="dec", help="Dec", type='string', default="")
parser.add_option("-m", "--params", dest="params",help="Input parameters yaml file",type='string',default="")
parser.add_option("-f", action="store_true", dest="force", help="Force reduction", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with -h robo43_pipeline.py")
    sys.exit(1)

# load pipeline parameters
p = s4params.load_sparc4_parameters()

# if reduced dir doesn't exist create one
if not os.path.exists(options.reducedir):
    os.mkdir(options.reducedir)

bias_list = sorted(glob.glob(options.bias))
flat_list = sorted(glob.glob(options.flat))
sci_list = sorted(glob.glob(options.science))

#print(bias_list)
#print(flat_list)
#print(sci_list)

p['master_bias'] = "{}/MasterZero.fits".format(options.reducedir)

flat_suffix = ""
if options.filter != "" :
    flat_suffix = "_{}".format(options.filter)
p['master_flat'] = "{}/MasterFlat{}.fits".format(options.reducedir,flat_suffix)

if 'MEM_CACHE_FOLDER' not in p.keys() or not os.path.exists(p['MEM_CACHE_FOLDER']) :
    p['MEM_CACHE_FOLDER'] = None
p['SOLAR_SYSTEM_OBJECT'] = False
p["READNOISEKEY"] = "GAIN"

try :
    obj_match_simbad, obj_coords = s4utils.match_object_with_simbad(options.object, search_radius_arcsec=10)
    object_id = obj_match_simbad["MAIN_ID"][0]

    p['SOLVE_ASTROMETRY_IN_STACK'] = True
    p['RA_DEG'] = obj_coords.ra.deg
    p['DEC_DEG'] = obj_coords.dec.deg
    
    print("Found Simbad match to object {} -> MAIN_ID: {}  RA: {}  DEC: {}".format(options.object,object_id,p['RA_DEG'],p['DEC_DEG']))
except :
    object_id = options.object
    p['SOLVE_ASTROMETRY_IN_STACK'] = False
    p['RA_DEG'] = 0.0
    p['DEC_DEG'] = 0.0
    print("WARNING: Could not find Simbad match to object {}".format(options.object))
        
p["ASTROM_REF_IMG"] = "/Users/eder/Data/ROBO43/ref_image_robo43.fits"
p['TIME_KEY'] = 'FRAME'
p['EXPTIMEKEY'] = 'EXPOSURE'
if options.time_key != "":
    p["TIME_KEY"] = options.time_key
# define number of files for stack
p['NFILES_FOR_STACK'] = 10
# define saturation limit
p['SATURATION_LIMIT'] = 65000
# define threshold for source detection
p['PHOT_THRESHOLD'] = 50

# calculate master bias and save product to fits
p = s4pipelib.run_master_calibration(p,
                                     inputlist=bias_list,
                                     output=p['master_bias'],
                                     obstype='bias',
                                     reduce_dir=options.reducedir,
                                     force=options.force,
                                     plot=options.plot)

# calculate master dome flat and save product to FITS
p = s4pipelib.run_master_calibration(p,
                                     inputlist=flat_list,
                                     output=p['master_flat'],
                                     obstype='flat',
                                     reduce_dir=options.reducedir,
                                     normalize=True,
                                     force=options.force,
                                     plot=options.plot)


if options.calibdbdir != "" :
    p["CALIBDB_DIR"] = options.calibdbdir

# set object id
object_id = options.object

# set suffix for stack product
stack_suffix = "{}_{}".format(object_id.replace(" ", ""),options.filter)


# calculate stack and reduce science images
p = s4pipelib.stack_and_reduce_sci_images(p,
                                          sci_list,
                                          options.reducedir,
                                          ref_img="",
                                          stack_suffix=stack_suffix,
                                          force=options.force,
                                          match_frames=True,
                                          polarimetry=False,
                                          plot=options.plot)

ts_suffix = "{}_{}".format(object_id.replace(" ", ""),options.filter)


if options.verbose:
    print("Start generating photometric time series products with suffix: ",ts_suffix)

list_of_catalogs = s4pipelib.get_list_of_catalogs(p['PHOT_APERTURES_FOR_LIGHTCURVES'], p['INSTMODE_PHOTOMETRY_KEYVALUE'])

# run photometric time series
phot_ts_product = s4pipelib.phot_time_series(p['OBJECT_REDUCED_IMAGES'][1:],
                                             ts_suffix=ts_suffix,
                                             reduce_dir=options.reducedir,
                                             time_key=p['TIME_KEYWORD_IN_PROC'],
                                             time_format=p['TIME_FORMAT_IN_PROC'],
                                             catalog_names=list_of_catalogs,
                                             time_span_for_rms=p['TIME_SPAN_FOR_RMS'],
                                             filter=options.filter,
                                             force=options.force)

target = p['TARGET_INDEX']
comps = p['COMPARISONS']

# Uncomment below to override deafult target/comparison definitions
target = 5
comps = [2, 3, 4, 6, 7]

plot_coords_file = phot_ts_product.replace(".fits","_coords{}".format(p['PLOT_FILE_FORMAT']))
plot_rawmags_file = phot_ts_product.replace(".fits","_rawmags{}".format(p['PLOT_FILE_FORMAT']))
plot_lc_file = phot_ts_product.replace(".fits",p['PLOT_FILE_FORMAT'])

s4plt.plot_light_curve(phot_ts_product,
                       target=target,
                       comps=comps,
                       nsig=10,
                       catalog_name=p['PHOT_REF_CATALOG_NAME'],
                       plot_coords=True,
                       plot_rawmags=True,
                       plot_sum=True,
                       plot_comps=True,
                       output_coords=plot_coords_file,
                       output_rawmags=plot_rawmags_file,
                       output_lc=plot_lc_file)
