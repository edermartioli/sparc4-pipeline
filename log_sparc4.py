# Description: Script to make a log of SPARC4 observations
# Author: Eder Martioli
# Laboratorio Nacional de Astrofisica, Brazil
# May 2022
#
# Example:
# python log_sparc4.py --input=*.fits --output=data.csv
#

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os, sys
import glob
import astropy.io.fits as fits
import numpy as np

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir", help="data directory",type='string',default="./")
parser.add_option("-i", "--input", dest="input", help="input data pattern",type='string',default="*.fits")
parser.add_option("-o", "--output", dest="output", help="output csv log file",type='string',default="")

parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with log_sparc4.py -h ")
    sys.exit(1)

if options.verbose:
    print('Data directory: ', options.datadir)
    print('Data input: ', options.input)
    print('Output csv log file: ', options.output)

currdir = os.getcwd()
os.chdir(options.datadir)

inputdata = sorted(glob.glob(options.input))

text = "IMG_INDEX/NIMGS,FILENAME,BASE_ID,CHANNEL_ID,SUFFIX,INSTRUME,OBSTYPE,OBJECT,FILTER,DATE,RA,DEC,EXPTIME,CAMFOC,DTEMP,EMMODE,READOUT,PREAMP,VSHIFT,READMODE,RDNOISE,GAIN,EMGAIN,CUBE_LENGTH,MEAN,MEDIAN,STD,MIN,MAX\n"

if options.output != "" :
    # open the file in the write mode
    f = open(options.output, 'w')

for i in range(len(inputdata)) :
    try :
        hdu = fits.open(inputdata[i])
        hdr = hdu[0].header
    
        basename = os.path.basename(inputdata[i])
    
        x = basename.split("_")
        base_id = x[0]
        channel_id = x[1][:2]
        suffix = ""
        for j in range(2,len(x)) :
            suffix += x[j]
            if j < len(x) - 1 :
                suffix += "_"
        suffix = suffix.replace(".fits","")
        if suffix == channel_id or suffix == "":
            suffix = "None"
            
        cube = hdu[0].data
        cube_length = len(cube)
    
        mean = np.nanmean(cube)
        median = np.nanmedian(cube)
        std = np.nanstd(cube)
        min = np.nanmin(cube)
        max = np.nanmax(cube)

        line_text = "{}/{},{},{},{},{},{},{},{},{},{},{},{},{:.1f},{},{},{},{},{},{},{},{},{},{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(i+1,len(inputdata),inputdata[i],base_id,channel_id,suffix,hdr["INSTRUME"],hdr["OBSTYPE"],hdr["OBJECT"],hdr["FILTER"],hdr["DATE"],hdr["RA"],hdr["DEC"],hdr["EXPTIME"],hdr["CAMFOC"],hdr["DTEMP"],hdr["EMMODE"],hdr["READOUT"],hdr["PREAMP"],hdr["VSHIFT"],hdr["READMODE"],hdr["RDNOISE"],hdr["GAIN"],hdr["EMGAIN"], cube_length,mean,median,std,min,max)
        
        text += line_text
        
    except :
        print("ERROR: could not read file:",inputdata[i])
        
if options.verbose :
    print(text)

if options.output != "" :
    f.write(text)
    # close the file
    f.close()

os.chdir(currdir)
