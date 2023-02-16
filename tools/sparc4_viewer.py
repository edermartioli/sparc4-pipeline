# -*- coding: iso-8859-1 -*-
"""
    Created on Jul 25 2022
    
    Description: Tool to view images in the 4 channels of SPARC4
    
    @author: Eder Martioli <martioli@lna.br>
    
    Laboratório Nacional de Astrofísica - LNA/MCTI
    
    Simple usage example:

    python /Volumes/Samsung_T5/sparc4-pipeline/tools/sparc4_seeing_monitor.py --datadir=


    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
from optparse import OptionParser

import astropy.io.fits as fits
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
import time

from astropop.photometry import background, starfind

#sparc4_pipeline_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir", help="data directory",type='string',default="./")
parser.add_option("-i", "--pattern", dest="pattern", help="data pattern",type='string',default="")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_monitor.py")
    sys.exit(1)

data_dir = options.datadir

filters = ["B","V","R","I"]

object_suffix = 'aumic'

# Set OPD geographic coordinates
longitude = -(45 + (34 + (57/60))/60) #hdr['OBSLONG']
latitude = -(22 + (32 + (4/60))/60) #hdr['OBSLAT']
altitude = 1864*u.m #hdr['OBSALT']
opd_location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=altitude)

inputdata = {}
filters_pos = {"B":(0,0),"V":(0,1),"R":(1,0),"I":(1,1)}
nrows, ncols = 2, 2
fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True,squeeze=True)

for k in range(10) :
    for i in range(len(filters)) :
        if options.pattern == "":
            inputdata[filters[i]] = glob.glob("{}/*{}_{}.fits".format(data_dir,object_suffix,filters[i]))
        else :
            inputdata[filters[i]] = glob.glob("{}_{}.fits".format(options.pattern,filters[i]))
           
        try :

            hdu = fits.open(inputdata[filters[i]][-1])
            data = hdu[0].data[0]

            #max = np.nanmax(data)
            date = hdu[0].header["DATE"]
            obstime = Time(date, format='isot', scale='utc', location=opd_location)
        except :
            continue
          
        row, col = filters_pos[filters[i]][0], filters_pos[filters[i]][1]

        #axs[row,col].clf()
        axs[row,col].set_title(r"Filter {} ({})".format(filters[i],obstime.isot))
        axs[row,col].imshow(data, vmin=np.median(data), vmax=np.percentile(data, 99.5))
            
    #plt.xlabel("Time")
    #plt.ylabel("FWHM (pixel)")
    #plt.legend()
    plt.show(block=False)

    time.sleep(100)


