"""
    Created on Apr 26 2024

    Description: Module to fix offending header

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python tools/sparc4_fix_header.py --datadir="/home/sparc4/data/" --nights="20240425" --pattern="*dflat.fits" -o --suffix="_mod" --keywords="OBJECT,OBSTYPE" --new_values="DOMEFLAT,FLAT" --dtypes="str,str"
    
    """

import os, sys
from optparse import OptionParser

import glob
import astropy.io.fits as fits

parser = OptionParser()
parser.add_option("-n", "--nights", dest="nights",help="Input list of night directories separated by comma", type='string', default="")
parser.add_option("-r", "--datadir", dest="datadir",help="Root data dir", type='string', default="")
parser.add_option("-p", "--pattern", dest="pattern",help="Pattern to select images", type='string', default="*.fits")
parser.add_option("-s", "--suffix", dest="suffix",help="Suffix to add in modified data", type='string', default="")
parser.add_option("-k", "--keywords", dest="keywords",help="Input list of keywords to update", type='string', default="")
parser.add_option("-e", "--new_values", dest="new_values",help="Input list of keyword values to upadte", type='string', default="")
parser.add_option("-t", "--dtypes", dest="dtypes",help="Data types: str, bool, int, or float", type='string', default="")
parser.add_option("-o", action="store_true", dest="overwrite",help="overwrite", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with -h sparc4_fix_header.py")
    sys.exit(1)

nights = options.nights.split(",")
keywords = options.keywords.split(",")
new_values = options.new_values.split(",")
dtypes = options.dtypes.split(",")

print("Requesting to fix the following keywords:")
for j in range(len(keywords)) :
    print("{} of {} : {} = {} (dtype={})".format(j+1,len(keywords),keywords[j],new_values[j],dtypes[j]))

for ch in range(1,5) :
    print("Starting fix header for CHANNEL {}".format(ch))

    for night in nights :
    
        night_dir = "{}/sparc4acs{}/{}".format(options.datadir,ch,night)
        night_pattern = "{}/{}".format(night_dir,options.pattern)
        
        inputdata = sorted(glob.glob(night_pattern))
        
        print("Night directory: {}".format(night_dir))
        print("Found {} images matching pattern: {} ".format(len(inputdata),night_pattern))

        for i in range(len(inputdata)) :
                
            try :
                outfile = inputdata[i].replace(".fits","{}.fits".format(options.suffix))
                
                hdul = fits.open(inputdata[i])
            
                for j in range(len(keywords)) :
                    new_value = new_values[j]
                    if dtypes[j] == "str" :
                        new_value = str(new_values[j])
                    elif dtypes[j] == "bool" :
                        new_value = bool(new_values[j])
                    elif dtypes[j] == "int" :
                        new_value = int(new_values[j])
                    elif dtypes[j] == "float" :
                        new_value = float(new_values[j])
                    
                    hdul[0].header[keywords[j]] = new_value
            
                print("{}/{} : src_file={} mod_file={}".format(i+1,len(inputdata),inputdata[i],outfile))
            
                hdul.writeto(outfile,overwrite=options.overwrite)
            except Exception as e:
                print("ERROR: could not fix header of image: {}".format(inputdata[i]))
                continue
        
