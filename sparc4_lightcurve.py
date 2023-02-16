# -*- coding: iso-8859-1 -*-
"""
    Created on Dec 1 2022
    
    Description: SPARC4 tool to calculate light curves
    
    @author: Eder Martioli <martioli@lna.br>
    
    Laboratório Nacional de Astrofísica - LNA/MCTI
    
    Simple usage example:

    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_lightcurve.py --input=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs2/20221112/20221112_s4c2_mls_*_proc.fits --output=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs2/20221112/ --target_index=4 --comparison_indices="0,1,2,5,6,7,9,10,11" -pv


* CHANNEL 1 :
    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_lightcurve.py --input=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs1/20221112/20221112_s4c1_mls*_proc.fits --output=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs1/20221112/ --target_index=5 --comparison_indices="0,1,2,3" -pv
* CHANNEL 2 :
    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_lightcurve.py --input=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs2/20221112/20221112_s4c2_mls*_proc.fits --output=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs2/20221112/ --target_index=4 --comparison_indices="0,1,2,3,5" -pv
* CHANNEL 3 :
    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_lightcurve.py --input=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs3/20221112/20221112_s4c3_mls*_proc.fits --output=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs3/20221112/ --target_index=6 --comparison_indices="0,1,2,3"
* CHANNEL 4 :
    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_lightcurve.py --input=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs4/20221112/20221112_s4c4_mls*_proc.fits --output=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced/sparc4acs4/20221112/ --target_index=6 --comparison_indices="1,2,3,4,5" -pv

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
from optparse import OptionParser

import sparc4_product_plots as s4plt
import sparc4_pipeline_lib as s4pipelib
import glob

sparc4_pipeline_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input pattern for reduced data",type='string',default="")
parser.add_option("-o", "--output", dest="output", help="Output lightcurve products",type='string',default="")
parser.add_option("-t", "--target_index", dest="target_index", help="Target index",type='int',default=0)
parser.add_option("-c", "--comparison_indices", dest="comparison_indices", help="Comparison indices",type='string',default="1,2")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_lightcurve.py")
    sys.exit(1)

inputdata = sorted(glob.glob(options.input))

if options.plot :
    s4plt.plot_sci_frame(inputdata[0], nstars=20)

target = options.target_index
comparison_indices = options.comparison_indices.split(",")
comps = []
for i in comparison_indices :
    comps.append(int(i))

# run photometric time series
phot_ts_product = s4pipelib.phot_time_series(inputdata, target=target, comps=comps, reduce_dir=options.output)

# plot light curve
s4plt.plot_light_curve(phot_ts_product, target=target, comps=comps, nsig=3, plot_sum=False, plot_comps=True)

