"""
    Created on Jun 4 2023

    Description: Mini pipeline to process SPARC4 data in photometric and polarimetric mode

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage examples:

    python -W ignore sparc4_mini_pipeline.py --nightdir=20230604 -vp

    python -W ignore sparc4_mini_pipeline.py --nightdir=20230604 --datadir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_jun23/ --reducedir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_jun23/reduced -v

    python -W ignore sparc4_mini_pipeline.py --nightdir=20230606 --datadir=/Volumes/Samsung_T5/Data/SPARC4/standards --reducedir=/Volumes/Samsung_T5/Data/SPARC4/standards/reduced -v
    
    python -W ignore scripts/sparc4_mini_pipeline.py --nightdir=20231107 --datadir=/Users/eder/Data/SPARC4/wasp-78/ --reducedir=/Users/eder/Data/SPARC4/wasp-78/reduced -v
    """

import os
import sys
from optparse import OptionParser

import sparc4.db as s4db
import sparc4.pipeline_lib as s4pipelib

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",help="SPARC4 channels: e.g '1,3,4' ", type='string',default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",help="Name of night directory common to all channels",type='string', default="")
parser.add_option("-f", action="store_true", dest="force",help="Force reduction", default=False)
parser.add_option("-p", action="store_true", dest="plot",help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_mini_pipeline.py")
    sys.exit(1)

match_frames = True
fit_zero_of_wppos = True

# initialize pipeline parameters
p = s4pipelib.init_s4_p(options.nightdir,
                        options.datadir,
                        options.reducedir,
                        options.channels,
                        print_report=options.verbose)

# Run full reduction for selected channels
for channel in p['SELECTED_CHANNELS']:

    # set zero based index of current channel
    j = channel - 1

    data_dir = p['data_directories'][j]
    ch_reduce_dir = p['ch_reduce_directories'][j]
    reduce_dir = p['reduce_directories'][j]

    # if db doesn't exist create one
    if not os.path.exists(p['s4db_files'][j]) or options.force :
        db = s4db.create_db_from_observations(p['filelists'][j], p['DB_KEYS'], include_img_statistics=p["INCLUDE_IMG_STATISTICS"],include_only_fullframe=p["FULL_FRAMES_ONLY"], output=p['s4db_files'][j])
    else:
        db = s4db.create_db_from_file(p['s4db_files'][j])

    # detect all detector modes
    detector_modes = s4db.get_detector_modes_observed(db, science_only=True, detector_keys=p["DETECTOR_MODE_KEYWORDS"])

    for key in detector_modes.keys():

        # create a list of zeros for current detector mode
        zero_list = s4db.get_file_list(db, obstype=p['BIAS_OBSTYPE_KEYVALUE'], detector_mode=detector_modes[key])
        # calculate master bias
        p["master_bias"] = "{}/{}_s4c{}{}_MasterZero.fits".format(reduce_dir, options.nightdir, p['CHANNELS'][j], key)
        p = s4pipelib.run_master_calibration(p, inputlist=zero_list, output=p["master_bias"], obstype='bias', data_dir=data_dir, reduce_dir=reduce_dir, force=options.force)

        # create a list of sky flats
        skyflat_list = s4db.get_file_list(
            db, detector_mode=detector_modes[key], skyflat=True)
        if len(skyflat_list):
            # calculate master sky flat
            p["master_skyflat"] = "{}/{}_s4c{}{}_MasterSkyFlat.fits".format(reduce_dir, options.nightdir, p['CHANNELS'][j], key)
            p = s4pipelib.run_master_calibration(p, inputlist=p["master_skyflat"], output=master_skyflat, obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=options.force)

        # create a list of flats for current detector mode
        flat_list = s4db.get_file_list(db, obstype=p['FLAT_OBSTYPE_KEYVALUE'], detector_mode=detector_modes[key])
        # calculate master dome flat
        p["master_flat"] = "{}/{}_s4c{}{}_MasterDomeFlat.fits".format(reduce_dir, options.nightdir, p['CHANNELS'][j], key)
        p = s4pipelib.run_master_calibration( p, inputlist=flat_list, output=p["master_flat"], obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=options.force)

        # set astrometry ref image as the one for this channel
        p["ASTROM_REF_IMG"] = os.path.join(p["CALIBDB_DIR"], p["ASTROM_REF_IMGS"][j])

        try:
            # reduce science data in photometric mode
            p = s4pipelib.reduce_sci_data(db, p, j, p['INSTMODE_PHOTOMETRY_KEYVALUE'], detector_modes[key], options.nightdir, reduce_dir, polar_mode=None, fit_zero=False, detector_mode_key=key, calw_mode="OFF", match_frames=match_frames, force=options.force, verbose=options.verbose, plot_stack=options.plot, plot_lc=options.plot, plot_polar=False)
        except Exception as e:
            print("WARNING: Could not reduce {} mode, detector mode {} : {}".format(p['INSTMODE_PHOTOMETRY_KEYVALUE'], key, e))
        
        try:
            # reduce science data in polarimetric L2 mode
            p = s4pipelib.reduce_sci_data(db, p, j, p['INSTMODE_POLARIMETRY_KEYVALUE'], detector_modes[key], options.nightdir, reduce_dir, polar_mode=p['POLARIMETRY_L2_KEYVALUE'], fit_zero=False, detector_mode_key=key, calw_mode=p['CALW_MODE'], match_frames=match_frames, force=options.force, verbose=options.verbose, plot_stack=options.plot, plot_lc=options.plot, plot_polar=p["PLOT_POLARIMETRY_FIT"])
        except Exception as e:
            print("WARNING: Could not reduce {}-{} mode, detector mode {} : {}".format(p['INSTMODE_POLARIMETRY_KEYVALUE'], p['POLARIMETRY_L2_KEYVALUE'], key, e))
        
        try:
            # reduce science data in  polarimetric L4 mode
            p = s4pipelib.reduce_sci_data(db, p, j, p['INSTMODE_POLARIMETRY_KEYVALUE'], detector_modes[key], options.nightdir, reduce_dir, polar_mode=p['POLARIMETRY_L4_KEYVALUE'], fit_zero=fit_zero_of_wppos, detector_mode_key=key, calw_mode=p['CALW_MODE'], match_frames=match_frames, force=options.force, verbose=options.verbose, plot_stack=options.plot, plot_lc=options.plot, plot_polar=p["PLOT_POLARIMETRY_FIT"])
        except Exception as e:
            print("WARNING: Could not reduce {}-{} mode, detector mode {} : {}".format(p['INSTMODE_POLARIMETRY_KEYVALUE'], p['POLARIMETRY_L4_KEYVALUE'], key, e))
        
