"""
    Created on Apr 30 2024

    Description: Module to run quick look

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python tools/sparc4_quickloook.py --nightdir=20230604
    
    """

import os, sys
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np

import sparc4.product_plots as s4plt
import sparc4.pipeline_lib as s4pipelib

import time
import itertools

def update_lightcurve_data(p, filename, nightdir, datadir="", reducedir="", params="", channels="1,2,3,4"):

    for cnt in itertools.count():
    
        # Run pipeline to generate new products
        command = "python -W ignore scripts/sparc4_mini_pipeline.py --nightdir={} --channels='{}' ".format(nightdir, channels)
        if datadir != "" :
            command += "--datadir={} ".format(datadir)
        if reducedir != "" :
            command += "--reducedir={} ".format(reducedir)
        if params != "" :
            command += "--params={} ".format(params)
        print("Running: ", command)
        #os.system(command)
    
        
        # check latest products:
        
        
    
    
        s4plt.plot_light_curve(filename, target=p['TARGET_INDEX'], comps=p['COMPARISONS'], nsig=10, plot_coords=False, plot_rawmags=False, plot_sum=True, plot_comps=True, catalog_name=p['PHOT_REF_CATALOG_NAME'])

        yield cnt
        

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",help="SPARC4 channels: e.g '1,3,4' ", type='string',default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",help="Name of night directory common to all channels",type='string',default="")
parser.add_option("-s", "--sleep_time", dest="sleep_time",help="Sleep time before re-running pipeline",type='int',default=5)
parser.add_option("-m", "--params", dest="params",help="Input parameters yaml file",type='string',default="")
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with -h sparc4_quickloook.py")
    sys.exit(1)

# initialize pipeline parameters
p = s4pipelib.init_s4_p(options.nightdir,
                        options.datadir,
                        options.reducedir,
                        options.channels,
                        print_report=options.verbose,
                        param_file=options.params)

lc_product = "/Users/eder/Data/SPARC4/wasp-78/reduced/sparc4acs1/20231107/20231107_s4c1_WASP-78b_POLAR_L2_S+N_lc.fits"

gene = update_lightcurve_data(p, lc_product, options.nightdir, datadir=options.datadir, reducedir=options.reducedir, params=options.params, channels=options.channels)

for i in gene :
    print("Iteration number: {} with sleep time of {}".format(i, options.sleep_time))
    time.sleep(options.sleep_time)
