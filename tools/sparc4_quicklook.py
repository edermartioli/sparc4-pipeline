"""
    Created on Apr 30 2024

    Description: Module to run quick look

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python tools/sparc4_quicklook.py --params=/Users/eder/Data/SPARC4/quicklook/sparc4_quicklook_params.yaml

    """

import os, sys
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np

import sparc4.pipeline_lib as s4pipelib

import time
import itertools

def run_sparc4_pipeline(nightdir, datadir, reducedir="", params="", channels="1,2,3,4",sync=False,obase="/mnt/win_sparc4acs"):

    channels_vector = channels.split(",")

    for cnt in itertools.count():
        if sync :
            qlrawdbase="{}/sparc4acs".format(datadir)
            for i in range(len(channels_vector)) :
                print("Syncing sparc4acs{} -> {}{}/".format(channels_vector[i], qlrawdbase, channels_vector[i]))
                #command = "rsync -avz --chmod=755 --progress {}{}/{} {}{}/".format(obase, channels_vector[i], nightdir, qlrawdbase, channels_vector[i])
                command = "rsync -avz --progress {}{}/{} {}{}/".format(obase, channels_vector[i], nightdir, qlrawdbase, channels_vector[i])
                print(command)
                os.system(command)
    
        start_time = time.time()
        # Run pipeline to generate new products
        command = "python -W ignore scripts/sparc4_mini_pipeline.py --nightdir={} --channels='{}' -pv ".format(nightdir, channels)
        if datadir != "" :
            command += "--datadir={} ".format(datadir)
        if reducedir != "" :
            command += "--reducedir={} ".format(reducedir)
        if params != "" :
            command += "--params={} ".format(params)
        print("Running: ", command)
        os.system(command)
    
        end_time = time.time()
        print(f"Quick look reduction finished in {end_time - start_time:.2f} seconds.")
        
        yield cnt
        

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",help="SPARC4 channels: e.g '1,3,4' ", type='string',default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",help="Name of night directory common to all channels",type='string',default="today")
parser.add_option("-t", "--sleep_time", dest="sleep_time",help="Sleep time before re-running pipeline",type='int',default=5)
parser.add_option("-m", "--params", dest="params",help="Input parameters yaml file",type='string',default="")
parser.add_option("-s", action="store_true", dest="sync",help="sync with remote data", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with -h sparc4_quickloook.py")
    sys.exit(1)

#obase="/mnt/win_sparc4acs"
obase="/Users/eder/Data/SPARC4/minidata_ql/sparc4acs"

# initialize pipeline parameters
p = s4pipelib.init_s4_p(options.nightdir,
                        options.datadir,
                        options.reducedir,
                        options.channels,
                        print_report=options.verbose,
                        param_file=options.params)

datadir = p['ROOTDATADIR']
reducedir = p['ROOTREDUCEDIR']

gene = run_sparc4_pipeline(options.nightdir, datadir=datadir, reducedir=reducedir, params=options.params, channels=options.channels, sync=options.sync, obase=obase)

for i in gene :
    print("Iteration number: {} with sleep time of {}".format(i, options.sleep_time))
    time.sleep(options.sleep_time)


