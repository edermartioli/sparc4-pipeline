# Description: Script to remove cube dimension
# Author: Eder Martioli
# Laboratorio Nacional de Astrofisica, Brazil
# 31 May 2022
#
# Example:
# python remove_cube_dimension.py --input=*.fits
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

from typing import Collection, Union

ExtensionHDU = Union[fits.ImageHDU, fits.BinTableHDU]
HDU = Union[fits.PrimaryHDU, ExtensionHDU]

def create_hdu_list(hdus: Collection[HDU]) -> fits.HDUList:
    """
    Takes a collection of fits HDUs and converts into an HDUList ready to be saved to a file
    :param hdus: The collection of fits HDUs
    :return: A fits HDUList ready to be saved to a file
    """
    hdu_list = fits.HDUList()
    for hdu in hdus:
        if len(hdu_list) == 0:
            hdu.header['NEXTEND'] = len(hdus) - 1
        else:
            hdu.header.remove('NEXTEND', ignore_missing=True)
        hdu_list.append(hdu)
    return hdu_list

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir", help="data directory",type='string',default="./")
parser.add_option("-i", "--input", dest="input", help="input data pattern",type='string',default="*.fits")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with remove_cube_dimension.py -h ")
    sys.exit(1)

if options.verbose:
    print('Data input: ', options.input)


currdir = os.getcwd()
os.chdir(options.datadir)

inputdata = sorted(glob.glob(options.input))

for i in range(len(inputdata)) :

    with fits.open(inputdata[i]) as hdu:
        shape = np.shape(hdu[0].data)
        dim3 = len(shape)
        print("Current sahpe:", shape)
        if dim3 > 2 :
            print("Removing 3rd dimension of image {}/{} -> {}".format(i+1,len(inputdata),inputdata[i]))
            # get header
            hdr  = hdu[0].header
            # create primary hdu with header of base image
            primary_hdu = fits.PrimaryHDU(header=hdr)
            # set data cube into primary extension
            primary_hdu.data = hdu[0].data[0]
            # create hdu list
            hdu_list = create_hdu_list([primary_hdu])
            # write image to fits
            hdu_list.writeto(inputdata[i], overwrite=True, output_verify="fix+warn")
        else :
            print("Image is already 2D {}/{} -> {}".format(i+1,len(inputdata),inputdata[i]))

os.chdir(currdir)

