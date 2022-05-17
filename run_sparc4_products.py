# -*- coding: iso-8859-1 -*-
"""
    Created on May 2 2022
    
    Description: This routine tests the sparc4_products.py library to produce SPARC4 products
    
    @author: Eder Martioli <martioli@lna.br>
    
    Laboratório Nacional de Astrofísica - LNA/MCTI
    
    Simple usage example:

    python run_sparc4_products.py

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
from optparse import OptionParser

import sparc4_products as s4p

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

from astropop.file_collection import FitsFileGroup
# astropop used modules
from astropop.image import imcombine, processing, imarith

import numpy as np

sparc4_pipeline_dir = os.path.dirname(__file__)
minidata_dir = os.path.join(sparc4_pipeline_dir, 'minidata/')
reduced_dir = os.path.join(sparc4_pipeline_dir, 'reduced/')
dbfilepath = os.path.join(minidata_dir, 'minidata/minidata.db')

master_bias = os.path.join(reduced_dir, 'master_bias.fits')
master_flat = os.path.join(reduced_dir, 'master_flat.fits')


def run_master_image_example(obstype='ZERO', method='median', output='', normalize=False) :
    
    # select FITS files in the minidata directory and build database
    main_fg = FitsFileGroup(location=minidata_dir, fits_ext=['.fits'], ext=0)
    # print total number of files selected:
    print(f'Total files: {len(main_fg)}')
    
    # Filter files by header keywords
    filter_fg = main_fg.filtered({'obstype': obstype})

    # print total number of bias files selected
    print(f'{obstype} files: {len(filter_fg)}')

    # get frames
    frames = list(filter_fg.framedata(unit='adu', use_memmap_backend=False))

    #extract gain from the first image
    gain = float(frames[0].header['GAIN'])*u.electron/u.adu  # using quantities is better for safety
    print('gain:', gain)

    # Perform gain calibration
    for i, frame in enumerate(frames):
        print(f'processing frame {i+1} of {len(frames)}')
        processing.gain_correct(frame, gain, inplace=True)

    # combine
    master = imcombine(frames, method=method, use_memmap_backend=False)

    # get statistics
    stats = master.statistics()
    
    norm_mean_value = master.mean()
    print('Normalization mean value:', norm_mean_value)

    data_units = 'electron'
    
    if normalize :
        master = imarith(master, norm_mean_value, '/')
        data_units = 'dimensionless'

    # write information into an info dict
    info = {'INCOMBME': ('{}'.format(method), 'imcombine method'),
        'INCOMBNI': (len(filter_fg), 'imcombine nimages'),
        'BUNIT': ('{}'.format(data_units), 'data units'),
        'DRSINFO': ('astropop', 'data reduction software'),
        'DRSROUT': ('master image', 'data reduction routine'),
        'NORMALIZ': (normalize, 'normalized master'),
        'NORMMEAN': (norm_mean_value.value,'normalization mean value in {}'.format(norm_mean_value.unit)),
        'MINVAL': (stats['min'].value,'minimum value in {}'.format(stats['min'].unit)),
        'MAXVAL': (stats['max'].value,'maximum value in {}'.format(stats['max'].unit)),
        'MEANVAL': (stats['mean'].value,'mean value in {}'.format(stats['mean'].unit)),
        'MEDIANVA': (stats['median'].value,'median value in {}'.format(stats['median'].unit)),
        'STDVALUE': (stats['std'].value,'standard deviation in {}'.format(stats['std'].unit))
        }
    
    # get data arrays
    img_data=np.array(master.data)[0]
    err_data=np.array(master.get_uncertainty())[0]
    mask_data=np.array(master.mask)[0]

    # call function masteZero from sparc4_products to generate final product
    mastercal = s4p.masterCalibration(filter_fg.files, img_data=img_data, err_data=err_data, mask_data=mask_data, info=info, filename=output)

    return mastercal


def make_catalog(frame) :

   catalog = {}
   
   catalog["star1"] = (1, 179.17943891, -0.38402173, 423.144, 469.8, 6.09, 6.34, 14.21, 0.1, 20.32, 0.1, 100, 0)
   catalog["star2"] = (2, 179.19232283, -0.3874216, 283.83, 513.0, 6.50, 6.80, 11.836, 0.1, 20.80, 0.1, 100, 0)

   return catalog
   

def run_science_image_example() :

    # load bias frame
    bias = s4p.getFrameFromMasterCalibration(master_bias)
    
    # load flat frame
    flat = s4p.getFrameFromMasterCalibration(master_flat)

    # select FITS files in the minidata directory and build database
    main_fg = FitsFileGroup(location=minidata_dir, fits_ext=['.fits'], ext=0)

    # Filter files by header keywords
    obj_fg = main_fg.filtered({'obstype': "OBJECT"})

    # print total number of bias files selected
    print(f'OBJECT files: {len(obj_fg)}')

    # get frames
    frames = list(obj_fg.framedata(unit='adu', use_memmap_backend=False))

    #extract gain from the first image
    gain = float(frames[0].header['GAIN'])*u.electron/u.adu  # using quantities is better for safety
    print('gain:', gain)

    data_units = 'electron'
    
    # write information into an info dict
    info = {'BUNIT': ('{}'.format(data_units), 'data units'),
        'DRSINFO': ('astropop', 'data reduction software'),
        'DRSROUT': ('science frame', 'data reduction routine'),
        'BIASSUB': (True, 'bias subtracted'),
        'BIASFILE': (master_bias, 'bias file name'),
        'FLATCORR': (True, 'flat corrected'),
        'FLATFILE': (master_flat, 'flat file name')
    }
    
    output_list = []
    
    # Perform gain calibration
    for i, frame in enumerate(frames):
        print(f'processing frame {i+1} of {len(frames)}')
        processing.gain_correct(frame, gain, inplace=True)
        processing.subtract_bias(frame, bias, inplace=True)
        processing.flat_correct(frame, flat, inplace=True)

        # get data arrays
        img_data=np.array(frame.data)[0]
        err_data=np.array(frame.get_uncertainty())[0]
        mask_data=np.array(frame.mask)[0]

        # make catalog
        catalog = make_catalog(frame)
    
        # get basename
        basename = os.path.basename(obj_fg.files[i])
        
        # create output name in the reduced dir
        output = os.path.join(reduced_dir, basename.replace(".fits","_proc.fits"))
        
        print(obj_fg.files[i],'->',output)
        # call function masteZero from sparc4_products to generate final product
        s4p.scienceImageProduct(obj_fg.files[i], img_data=img_data, err_data=err_data, mask_data=mask_data, info=info, catalog=catalog, filename=output)

        output_list.append(output)
        
    return output_list
    
    
def run_phot_time_series_example(sci_list) :

    # read photometric time series data from a list of SPARC4 science image products
    tsdata = s4p.readPhotTimeSeriesData(sci_list)
    
    # get header of first image in the time series
    hdr = fits.getheader(sci_list[0])
    
    # get object name
    objectname = hdr["OBJECT"].replace(" ","")
    
    # set output light curve product file name
    output = os.path.join(reduced_dir, "{}_lc.fits".format(objectname))
    
    # Construct information dictionary to add to the header of FITS product
    info = {}

    info['OBSERV'] = ('OPD', 'observatory')
    longitude = -(45 + (34 + (57/60))/60)
    latitude = -(22 + (32 + (4/60))/60)
    info['OBSLAT'] = (latitude, '[DEG] observatory latitude (N)')
    info['OBSLONG'] = (longitude, '[DEG] observatory longitude (E)')
    info['OBSALT'] = ('1864', '[m] observatory altitude')
    info['TELESCOP'] = ('OPD-PE 1.6m', 'telescope')
    info['INSTRUME'] = ('SPARC4', 'instrument')
    info['OBJECT'] = (hdr["OBJECT"], 'ID of object of interest')
    equinox = 'J2000.0'
    source = SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg), frame='icrs', equinox=equinox)
    info['RA'] = (source.ra.value, '[DEG] RA of object of interest')
    info['DEC'] = (source.dec.value, '[DEG] DEC of object of interest')
    info['RADESYS'] = ('ICRS    ', 'reference frame of celestial coordinates')
    info['EQUINOX'] = (2000.0, 'equinox of celestial coordinate system')
    info['PHZEROP'] = (0., '[mag] photometric zero point')
    info['PHOTSYS'] = ("SPARC4", 'photometric system')
    info['PHBAND'] = (hdr["FILTER"], 'photometric band pass')
    info['WAVELEN'] = (550., 'band pass central wavelength [nm]')
    info['BANDWIDT'] = (300., 'photometric band width [nm]')

    # generate the photometric time series product
    s4p.photTimeSeriesProduct(tsdata["TIME"], tsdata["RA"], tsdata["DEC"], tsdata["MAG"], tsdata["EMAG"], tsdata["SKYMAG"], tsdata["ESKYMAG"], tsdata["DMAG"], tsdata["EDMAG"], tsdata["FLAG"], info=info, filename=output)
    
    return output
    
    
parser = OptionParser()
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h run_sparc4_products.py")
    sys.exit(1)

# if reduced dir doesn't exist create one
if not os.path.exists(reduced_dir) :
    os.mkdir(reduced_dir)

# run master bias example
run_master_image_example(obstype='ZERO', method='median', output=master_bias)

# run master flat example
run_master_image_example(obstype='FLAT', method='median', output=master_flat, normalize=True)

# run science image example
sci_list = run_science_image_example()

# run photometric time series example
run_phot_time_series_example(sci_list)
