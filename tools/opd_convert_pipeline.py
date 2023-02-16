# -*- coding: iso-8859-1 -*-
"""
    Created on June 26 2022
    
    Description: Pipeline to convert data from OPDAcquisition
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/sparc4-pipeline/tools/opd_convert_pipeline.py --object_prefixes="aumic,TOI-4339"
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import astropy.io.fits as fits
import glob
from copy import deepcopy

parser = OptionParser()
parser.add_option("-b", "--bias_prefix", dest="bias_prefix", help="Input BIAS prefix",type='string',default="ZERO")
parser.add_option("-f", "--flat_prefix", dest="flat_prefix", help="Input FLAT prefix",type='string',default="FLAT")
parser.add_option("-o", "--object_prefixes", dest="object_prefixes", help="Input OBJECT prefixes",type='string',default="")
parser.add_option("-d", "--sparc4_pipeline_dir", dest="sparc4_pipeline_dir", help="SPARC4 pipeline directory",type='string',default="/Volumes/Samsung_T5/sparc4-pipeline/")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h opd_convert_pipeline.py")
    sys.exit(1)

if options.verbose:
    print('Spectral e.fits data pattern: ', options.input)
    print('LSD mask: ', options.lsdmask)

#sparc4_pipeline_dir = os.path.dirname(__file__) + '/'
sparc4_pipeline_dir = options.sparc4_pipeline_dir

command = "python {}convert_opdacq_to_sparc.py --object=ZERO --obstype=ZERO --input={}*.fits".format(sparc4_pipeline_dir, options.bias_prefix)
print("Running: ",command)
os.system(command)

command = "python {}convert_opdacq_to_sparc.py --object=FLAT --obstype=FLAT --input={}*.fits".format(sparc4_pipeline_dir, options.flat_prefix)
print("Running: ",command)
os.system(command)

objects = options.object_prefixes.split(",")

for objprefix in objects :
    command = "python {}convert_opdacq_to_sparc.py --obstype=OBJECT --input={}*.fits".format(sparc4_pipeline_dir,objprefix)
    print("Running: ",command)
    os.system(command)
