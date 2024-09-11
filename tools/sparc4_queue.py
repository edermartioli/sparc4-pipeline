"""
    Created on Apr 22 2024

    Description: Module to run a queue of nights

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python tools/sparc4_queue.py --nights="20230604,20230605,20230606"
    
    """

import os, sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--nights", dest="nights",help="Input list of night directories separated by comma", type='string', default="")
parser.add_option("-s", action="store_true", dest="sync",help="sync", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with -h sparc4_queue.py")
    sys.exit(1)

obase="/mnt/win_sparc4acs"
dbase="/home/sparc4/data/sparc4acs"

nights = options.nights.split(",")

for night in nights :

    if options.sync :
        for ch in range(4) :
            print("Syncing sparc4acs{}".format(ch+1))
            
            command = "rsync -avz --chmod=755 --progress {}{}/{} {}{}/".format(obase,ch+1,night,dbase,ch+1)
            os.system(command)
            
    command = "python -W ignore scripts/sparc4_mini_pipeline.py --nightdir={} -v".format(night)
    print("Running: ", command)
    os.system(command)
    
