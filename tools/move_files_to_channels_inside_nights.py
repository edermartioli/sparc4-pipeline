"""
    Created on Apr 22 2024

    Description: Module to run a queue of nights

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python tools/move_files_to_channels_inside_nights.py --datadir=/Users/eder/Data/SPARC4/minidata
    
    """

import os, sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",help="Input root data dir", type='string', default="")
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with -h move_files_to_channels_inside_nights.py")
    sys.exit(1)

datadir = options.datadir

nights = os.listdir("{}/sparc4acs1/".format(datadir))

for night in nights :
    for ch in range(1,5) :
        source_dir = "{}/sparc4acs{}/{}/".format(datadir,ch,night)
        target_dir = "{}/{}/sparc4acs{}/".format(datadir,night,ch)
        
        if os.path.exists(source_dir) :
            print("moving data from {} -> {}".format(source_dir,target_dir))
            # if target dir doesn't exist create one
            os.makedirs(target_dir, exist_ok=True)
            command = "mv {}/* {}/".format(source_dir,target_dir)
            print("Running command: ", command)
            os.system(command)
