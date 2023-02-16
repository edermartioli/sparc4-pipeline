# Description: Script to make a log of SPARC4 observations
# Author: Eder Martioli
# Laboratorio Nacional de Astrofisica, Brazil
# 31 May 2022
#
# Example:
# python /Volumes/Samsung_T5/sparc4-pipeline/convert_opdacq_to_sparc.py --object=HATS-20 --obstype=OBJECT --input=HATS-20*.fits
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

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation

from astroquery.simbad import Simbad

from copy import deepcopy
import sparc4_products as s4p


def convert_str2float_with_comma(input_string) :
    s,f = input_string.split(",")
    f = "0."+f
    outputfloat = float(s) + float(f)
    return outputfloat

def covert_to_sparc4_header(ohdr, obstype="", force_simbad_coords=False, object="", filter="") :

    # create empty header
    hdr = fits.PrimaryHDU().header

    # Below it needs to be tested what are cols and lines
    x1,x2,y2,y1 = ohdr["IMGRECT"].split(",")
    
    # calculate exptime:
    exptime = convert_str2float_with_comma(ohdr["EXPOSURE"])
    # calculate ccd temperature:
    try :
        ccdtemp = convert_str2float_with_comma(ohdr["TEMP"])
    except :
        ccdtemp = 'UNKNOWN'

    # calculate ambient temperature:
    ambienttemp = convert_str2float_with_comma(ohdr["W-TEMP"])
    # calculate ambient pressure:
    ambientpress = convert_str2float_with_comma(ohdr["W-BAR"])

    tempst = 'TEMPERATURE_UNSTABILIZED'
    if ohdr["UNSTTEMP"] == '-999    ' :
        tempst = 'TEMPERATURE_STABILIZED'

    # filter information not reliable in OPDACQ, so one can input a new value
    if filter == "" :
        filter = ohdr["FILTER"]

    #  one can input a new object name
    if object == "" :
        object = ohdr["OBJECT"]

    # Set OPD geographic coordinates
    longitude = -(45 + (34 + (57/60))/60) #hdr['OBSLONG']
    latitude = -(22 + (32 + (4/60))/60) #hdr['OBSLAT']
    altitude = 1864*u.m #hdr['OBSALT']
    opd_location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=altitude)

    ### Object coordinates
    ra, dec = None, None
    if 'DEC' in ohdr.keys() and force_simbad_coords == False :
        dec = ohdr['DEC']
    else :
        try :
            result_table = Simbad.query_object(object)
            dd,mm,ss = result_table["DEC"].value[0].split(" ")
            dec = "{}:{}:{}".format(dd,mm,ss)
        except :
            if 'DEC' in ohdr.keys() :
                dec = ohdr['DEC']
            else :
                dec = 'UNKNOWN'
        
    if 'RA' in ohdr.keys() and force_simbad_coords == False :
        ra = ohdr['RA']
    else :
        try :
            result_table = Simbad.query_object(object)
            dd,mm,ss = result_table["RA"].value[0].split(" ")
            ra = "{}:{}:{}".format(dd,mm,ss)
        except :
            if 'RA' in ohdr.keys() :
                ra = ohdr['RA']
            else :
                ra = 'UNKNOWN'

    equinox="J{:.1f}".format(float(ohdr["EPOCH"]))
    #equinox="{:.1f}".format(float(ohdr["EPOCH"]))

    # set source observed
    source = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs', equinox=equinox)

    # set obstime ** Using jd from TCS, rather use system's time
    obstime=Time(ohdr["DATE-OBS"], format='isot', scale='utc', location=opd_location)
    jd = obstime.jd
    mjd = obstime.mjd
    
    # Set light travel time for source observed
    ltt_bary = obstime.light_travel_time(source)
    bjd = obstime.tdb.jd + ltt_bary
    
    #### HJD
    ltt_helio = obstime.light_travel_time(source, 'heliocentric') ### para o HJD
    hjd = obstime.utc + ltt_helio
    
    # populate header as in the SPARC4 instrument
    hdr.set("SIMPLE",ohdr["SIMPLE"],"file does conform to FITS standard")
    hdr.set("BITPIX",ohdr["BITPIX"],"number of bits per data pixel")
    hdr.set("NAXIS",3,"number of data axes")
    hdr.set("NAXIS1",1024,"length of data axis")
    hdr.set("NAXIS2",1024,"length of data axis")
    hdr.set("NAXIS3",1,"length of data axis")
    hdr.set("EXTEND",True,"FITS dataset may contain extensions")
    hdr.set("BZERO",ohdr["BZERO"],"offset data range to that of unsigned short")
    hdr.set("BSCALE",ohdr["BSCALE"],"default scaling factor")
    hdr.set("FILENAME",ohdr["IMAGE"],"File name at acquisition")
    hdr.set("NIGHTDIR",None,"Night directory name")
    hdr.set("PROGRID",None,"Program ID")

    hdr.set("OBSTYPE",obstype,"Image type: OBJECT, BIAS, FLAT, DARK")
    hdr.set("OBJECT",object,"Object name")

    hdr.set("OBSERVER",ohdr["OBSERVER"],"Name of the observer")
    hdr.set("INSTRUME",ohdr["INSTRUME"],"Instrument")
    hdr.set("CHANNEL",None,"Instrument channel")

    hdr.set("LOCTIME",None,"LT date at start of exposure")
    hdr.set("DATE",ohdr["DATE-OBS"],"Aquisition UT date (YY/MM/DD:HH:MM:SS.SS)")
    hdr.set("UTDATE",ohdr["DATE-OBS"],"UT date at start of exposure")
    hdr.set("JD",jd,"Julian Day")
    hdr.set("MJD",mjd,"Modified Julian Date at start of exposure")
    hdr.set("EXPTIME",exptime,"Exposure time (s)")

    hdr.set("HBIN",int(ohdr["HBIN"]),"Horizontal binning")
    hdr.set("VBIN",int(ohdr["VBIN"]),"Vertical binning")
    hdr.set("INITCOL",int(x1),"Initial Column")
    hdr.set("FINALCOL",int(x2),"Final Column")
    hdr.set("INITLIN",int(y1),"Initial Line")
    hdr.set("FINALLIN",int(y2),"Final Line")

    hdr.set("ACQMODE",ohdr["ACQMODE"],"Acquisition Mode")
    hdr.set("READMOD",None,"Read out mode: fast, normal, slow")
    hdr.set("READMODE",ohdr["READMODE"],"CCD read mode")
    hdr.set("READOUT",None,"Readout Rate")
    hdr.set("VSHIFT",3,"Vertical Shift Speed")
    hdr.set("PREAMP","Gain {}".format(ohdr["PREAMP"].replace("x","").replace(" ","")),"Pre-amplifier gain")
    hdr.set("EMMODE",ohdr["OUTPTAMP"],"Output amplifier mode")
    hdr.set("EMGAIN",int(ohdr["EMREALGN"]),"EM Gain")
    hdr.set("GAIN",float(ohdr["GAIN"]),"Gain (e-/ADU)")
    hdr.set("GAINERR",0.0,"Gain error (e-/ADU)")
    hdr.set("RDNOISE",float(ohdr["RDNOISE"]),"Read noise (e-)")
    hdr.set("RNOISERR",0.0,"Read noise error (e-)")

    hdr.set("TRIGGER",ohdr["TRIGGER"],"Trigger Mode")
    hdr.set("SHUTTER",None,"Shutter Mode (OPEN, CLOSED, or AUTO)")
    hdr.set("OPENSHT",50,"Time to open the shutter (ms)")
    hdr.set("CLOSESHT",50,"Time to close the shutter (ms)")
    
    hdr.set("NEXPSEQ",5,"Total number of exposures in sequence")
    hdr.set("SKYFIBER",None,"Sky fiber: ON or OFF")
    
    hdr.set("KSERLEN",1,"Kinetic Series Length")
    hdr.set("SEQINDEX",4,"Exposure index in the sequence")
    # Index versus real value (?)
    hdr.set("HAD",0.0,"Hour Angle (deg)")
    
    hdr.set("AIRMASS",float(ohdr["AIRMASS"]),"Mean airmass for the observation")
    hdr.set("AMSTART",0.0,"Airmass at start of exposure")
    hdr.set("AMEND",0.0,"Airmass at end of exposure")
    hdr.set("CALFIBER",None,"Calibration Fiber: ON or OFF")
    
    hdr.set("EPOCH",float(ohdr["EPOCH"]),"Epoch for Target coordinates")
    hdr.set("RA_DEG",source.ra.value,"Requested Right Ascension (deg)")
    hdr.set("DEC_DEG",source.dec.value,"Requested Declination (deg)")
    hdr.set("TELRA",ra,"TCS right ascension: HH:MM:SS.SSS")
    hdr.set("TELDEC",dec,"TCS declination: +-DD:MM:SS.SSS")
    hdr.set("RA",ra,"Requested right Ascension: HH:MM:SS.SSS")
    hdr.set("DEC",dec,"Requested Declination: +- DD:MM:SS.SSS")
    hdr.set("TELFOC",float(ohdr["FOCUSVAL"]),"Telescope focus position (mm)")
    hdr.set("TCSUT",None,"TCS Universal Time")
    hdr.set("TCSST",ohdr["ST"],"TCS Sideral Time")
    hdr.set("CAMFOC",None,"Camera focus position (mm)")
    hdr.set("HA",ohdr["HA"],"Hour Angle (Sexagesimal)")

    hdr.set("XOFFSET",0.0,"Telescope RA offset in arcsec")
    hdr.set("YOFFSET",0.0,"Telescope DEC offset in arcsec")
    
    hdr.set("PA",0.0,"Current Rotator Position Angle (degree)")
    hdr.set("MOONDIST",0.0,"Moon angular distance to object (deg)")
    hdr.set("MOONALT",0.0,"Moon elevation above horizon (deg)")
    hdr.set("ADCSPEED",0.0,"ADC motor speed (rpm)")
    hdr.set("GUIDTEXP",0.0,"Guider exposure time (s)")
    hdr.set("GUIDFREQ",0.0,"Guiding frequency (Hz)")
    hdr.set("GUIDOBJX",0.0,"Guiding object x position (pixel)")
    hdr.set("GUIDOBJY",0.0,"Guiding object y position (pixel)")
    hdr.set("MOONPHAS",None,"Moon elevation above horizon (deg)")
    hdr.set("AVGYCORR",0.0,"Average correction in RA (arcsec)")
    hdr.set("ADC",None,"Atmospheric Dispersion Corrector: ON or OFF")
    hdr.set("AVGXCORR",0.0,"Average correction in DEC (arcsec)")
    hdr.set("CALMIRR",None,"Calibration mirror position: IN or OUT")
    hdr.set("GFOCUS",0.0,"Guider focus position (mm)")
    hdr.set("GUIDING",None,"Guider status: ON or OFF")
    hdr.set("DPOS",0.0,"Dome position (deg)")
    hdr.set("THARLAMP",None,"Thorium-Argon lamp state: ON or OFF")
    hdr.set("DTEMP",0.0,"Dome temperature (deg C)")
    hdr.set("HALLAMP",None,"Halogen lamp state: ON or OFF")
    hdr.set("WINSPEED",0.0,"Wind Speed (km/h)")
    hdr.set("TEMP",ccdtemp,"CCD temperature")
    hdr.set("AGITATOR",None,"Fiber agitator: ON or OFF")
    hdr.set("SERN",int(ohdr["SERNO"]),"Serial Number")
    hdr.set("THARMIRR",None,"Th-Ar mirror position: SPECTROGRAPH or TELESCOP")
    hdr.set("DSTATUS",None,"Dome Status: OPEN or CLOSED")
    hdr.set("DHUM",None,"Dome humidity (per cent)")
    hdr.set("ADCHNNL",0,"Analogical to Digital Channel")
    hdr.set("DLAMP",None,"Dome lamp: ON or OFF")
    hdr.set("DFLAT",None,"Dome flat lamp: ON or OFF")
    hdr.set("TEMPEXT",ambienttemp,"External temperature (deg C)")
    hdr.set("PRESSURE",ambientpress,"Pressure (mbar)")
    hdr.set("HUMIDITY",float(ohdr["W-HUM"]),"Humidity (per cent)")
    hdr.set("WINDDIR",None,"Wind direction: North-eastward (deg)")
    hdr.set("SHTTTL","TTL Low","Shutter TTL")
    hdr.set("COOLER","UNKNOWN","CCD cooler: ON or OFF")
    hdr.set("TEMPST",tempst,"Temperature status")
    hdr.set("FRAME_T","UNKNOWN","Frame Transfer: ON or OFF")
    hdr.set("VCLOCK",0,"Vertical Clock Amplitude: 0, +1, +2, +3, +4")
    hdr.set("SD","","Sideral Day")
    hdr.set("FILTER",filter,"The filter used in the observation: UBVRI")
    hdr.set("ACSVRSN","v1.10","The version of the ACS")
    hdr.set("CTRLINTE","OPDAcquisition","Graphical control interface")
    hdr.set("IM_DT0",0,"Delay in nsec of the image 1")

    # add keywords that are not yet in the SPARC4
    hdr.set("TELESCOP",ohdr["TELESCOP"],"Telescope")
    hdr.set("PLATESCL",float(ohdr["PLATESCL"]),'Pixel scale (arcsec/pixel)')
    hdr.set("OBSLONG",longitude,'Observatory East Longitude (DEG, 0 to 360)')
    hdr.set("OBSLAT",latitude,'Observatory North Latitude (DEG, -90 to 90)')
    hdr.set("OBSALT",1864.0,"Observatory elevation above sea level (m)")
    hdr.set("EQUINOX",float(ohdr["EPOCH"]),'Equinox')
    hdr.set("BJD",bjd.value,'Barycentric Julian Date')
    #hdr.set("HJD",hjd.value,'Heliocentric Julian Date')
    
    # Sugestões:
    # Agrupar os keywords relacionados. Por exemplo:
    # Keywords do TCS virem todos juntos
    # Agrupar parâmetros relacionados do CCD, e.g. HBIN, VBIN
    # Agrupar parâmetros de tempo, coordenadas, informações climáticas, instrumento, etc
    
    return hdr

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir", help="data directory",type='string',default="./")
parser.add_option("-i", "--input", dest="input", help="input data pattern",type='string',default="*.fits")
parser.add_option("-o", "--object", dest="object", help="Object name",type='string',default="")
parser.add_option("-t", "--obstype", dest="obstype", help="OBSTYPE",type='string',default="")
parser.add_option("-f", "--filter", dest="filter", help="Filter",type='string',default="")
parser.add_option("-s", "--simbad_coords", action="store_true", dest="simbad_coords", help="Force SIMBAD coords",default=False)
parser.add_option("-c", "--cube", action="store_true", dest="cube", help="cube",default=False)
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

for i in range(len(inputdata)) :

    print("Converting image {}/{} -> {}".format(i+1,len(inputdata),inputdata[i]))

    with fits.open(inputdata[i], mode='update') as hdu:
    #with fits.open(inputdata[i]) as hdu:

        hdr = covert_to_sparc4_header(hdu[0].header, obstype=options.obstype, force_simbad_coords=options.simbad_coords, object=options.object, filter=options.filter)
        
        if options.cube :
            # get image data
            img_data = hdu[0].data
            # create primary hdu with header of base image
            primary_hdu = fits.PrimaryHDU(header=hdr)
            # set data cube into primary extension
            primary_hdu.data = np.array([img_data])
            # create hdu list
            hdu_list = s4p.create_hdu_list([primary_hdu])
            # write image to fits
            hdu_list.writeto(inputdata[i], overwrite=True, output_verify="fix+warn")
        else :
            hdu[0].header = hdr
            hdu.flush()  # changes are written back to original fits

os.chdir(currdir)

