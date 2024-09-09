"""
    Created on Sep 8 2024

    Description: Module to fix offending header

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python tools/sparc4_fix_wppos.py --datadir="/home/sparc4/data/" --night="20240425" --firstimage="" --object="" --ncycles=1 --nframes=1 --channel=1
    
    
    """

import os, sys
from optparse import OptionParser

import glob
import astropy.io.fits as fits

parser = OptionParser()
parser.add_option("-n", "--night", dest="night",help="Input night", type='string', default="")
parser.add_option("-r", "--datadir", dest="datadir",help="Root data dir", type='string', default="")
parser.add_option("-1", "--firstimage", dest="firstimage",help="first image", type='string', default="")
parser.add_option("-o", "--object", dest="object",help="Object", type='string', default="")
parser.add_option("-c", "--ncycles", dest="ncycles",help="Number of cycles to fix", type='int', default=1)
parser.add_option("-f", "--nframes", dest="nframes",help="Number of frames per position", type='int', default=1)
parser.add_option("-p", "--npos", dest="npos",help="Number of waveplate positions", type='int', default=16)
parser.add_option("-j", "--channel", dest="channel",help="SPARC4 channel", type='int', default=1)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with -h sparc4_fix_wppos.py")
    sys.exit(1)

night = options.night
ch = options.channel
    
night_dir = "{}/sparc4acs{}/{}".format(options.datadir,ch,night)

night_pattern = "{}/*.fits".format(night_dir)
inputdata = sorted(glob.glob(night_pattern))
 
print("Night directory: {}".format(night_dir))

ii = 0
for i in range(len(inputdata)) :
    if os.path.basename(inputdata[i]) == os.path.basename(options.firstimage) :
        ii = i
        break

ncycles = options.ncycles
nframes = options.nframes
npos = options.npos

selected_data = inputdata[ii : ii + nframes * npos * ncycles]

wppos = []
for j in range(ncycles) :
    wp = 1
    for k in range(npos) :
        for i in range(nframes) :
            wppos.append(wp)
        wp+=1

for i in range(len(selected_data)) :
    #print(selected_data[i], wppos[i])
    try :
        hdul = fits.open(selected_data[i])
        if wppos[i] != hdul[0].header['WPPOS'] and hdul[0].header['WPPOS'] == 0 :
            print("Updating WPPOS value of image {}: {}->{}".format(os.path.basename(selected_data[i]),hdul[0].header['WPPOS'],wppos[i]))
            hdul[0].header['WPPOS'] = wppos[i]
            hdul.writeto(selected_data[i],overwrite=True)
    except Exception as e:
        print("ERROR: could not fix header of image: {}".format(selected_data[i]))
        continue

