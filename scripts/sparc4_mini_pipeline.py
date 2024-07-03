"""
    Created on Jun 4 2023

    Description: Mini pipeline to process SPARC4 data in photometric and polarimetric mode

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage examples:

    python -W ignore scripts/sparc4_mini_pipeline.py --nightdir=20230604 -vp --params=/Users/eder/sparc4-pipeline/params/my_params.yaml

    python -W ignore scripts/sparc4_mini_pipeline.py --nightdir=20230605 --datadir=/Users/eder/Data/SPARC4/comissioning_jun23/ --reducedir=/Users/eder/Data/SPARC4/comissioning_jun23/reduced -v

    python -W ignore sparc4_mini_pipeline.py --nightdir=20230606
    
    python -W ignore scripts/sparc4_mini_pipeline.py --nightdir=20231107 --datadir=/Users/eder/Data/SPARC4/wasp-78/ --reducedir=/Users/eder/Data/SPARC4/wasp-78/reduced -v
    """

import os
import sys
from optparse import OptionParser

import sparc4.db as s4db
import sparc4.pipeline_lib as s4pipelib
import sparc4.utils as s4utils

from copy import deepcopy

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",help="SPARC4 channels: e.g '1,3,4' ", type='string',default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",help="Name of night directory common to all channels",type='string',default="")
parser.add_option("-o", "--object", dest="object",help="Object ID (same as in header key OBJECT) to restrict reduction",type='string',default="")
parser.add_option("-m", "--params", dest="params",help="Input parameters yaml file",type='string',default="")
parser.add_option("-t", "--target_list", dest="target_list",help="Input target list",type='string',default="")
parser.add_option("-f", action="store_true", dest="force",help="Force reduction", default=False)
parser.add_option("-p", action="store_true", dest="plot",help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose",help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_mini_pipeline.py")
    sys.exit(1)


# initialize pipeline parameters
p = s4pipelib.init_s4_p(options.nightdir,
                        options.datadir,
                        options.reducedir,
                        options.channels,
                        print_report=options.verbose,
                        param_file=options.params)

# Run full reduction for selected channels
for channel in p['SELECTED_CHANNELS']:

    # set zero based index of current channel
    j = channel - 1

    data_dir = p['data_directories'][j]
    reduce_dir = p['reduce_directories'][j]

    # create output night log
    night_log_path = os.path.join(reduce_dir,"{}_sparc4acs{}.log".format(options.nightdir, p['CHANNELS'][j]))
    
    # create a night logger
    logger = s4utils.start_logger(night_log_path)
    
    # log messages:
    logger.info("Start logging, saving log to file: {}".format(night_log_path))
    logger.info("Reducing night: {}".format(options.nightdir))
    logger.info("Starting reduction for channel: {} of list {}".format(channel,p['SELECTED_CHANNELS']))
    logger.info("Raw data directory (source): {}".format(data_dir))
    logger.info("Reduction data directory (target): {}".format(reduce_dir))

    # create output night report file name
    night_report_file_path = os.path.join(reduce_dir,"{}_sparc4acs{}_night_report.txt".format(options.nightdir, p['CHANNELS'][j]))
      
    # log messages:
    logger.info("Writing night report to file: {}".format(night_report_file_path))

    # create a night report
    s4pipelib.write_night_report(p, night_report_file_path, channel_index=j)
    
    # if db doesn't exist create one
    if not os.path.exists(p['s4db_files'][j]) or options.force or p['FORCE_DB_CREATION'] :
        # log messages:
        logger.info("Creating new database file: {}".format(p['s4db_files'][j]))
        
        db = s4db.create_db_from_observations(p['filelists'][j], p['DB_KEYS'], include_img_statistics=p['INCLUDE_IMG_STATISTICS'],include_only_fullframe=p['FULL_FRAMES_ONLY'], valid_obstype_keys=p['OBSTYPE_VALID_KEYVALUES'], mandatory_keys=p['MANDATORY_KEYWORDS'], output=p['s4db_files'][j])
    else:
        # log messages:
        logger.info("Loading database from existing file: {}".format(p['s4db_files'][j]))
        
        db = s4db.create_db_from_file(p['s4db_files'][j])
    
    if db is None :
        # log messages:
        logger.warn("No files detected for reduction of channel {}, skipping channel.".format(channel))
        continue
        
    # Create target list file name for a given night and channel
    p["TARGET_LIST_FILE"] = os.path.join(reduce_dir,"{}_sparc4acs{}_target_list.csv".format(options.nightdir, p['CHANNELS'][j]))

    if options.target_list != "" :
        # log messages:
        logger.info("Loading target list from input file: {}".format(options.target_list))

        # Copy input target list file to reduction location using a standard file name
        print("Copying input target list file {} to {}".format(options.target_list, p["TARGET_LIST_FILE"]))
        command = "cp {} {} ".format(options.target_list,p["TARGET_LIST_FILE"])
        os.system(command)
        
    # log messages:
    logger.info("Generating target list and saving to file: {}".format(p["TARGET_LIST_FILE"]))
    # Add targets that has a SIMBAD match with either the OBJECT or RA/DEC key values
    target_list = s4pipelib.build_target_list_from_data(p['objects'][j], p['obj_skycoords'][j], search_radius_arcsec=p["COORD_SEARCH_RADIUS_IN_ARCSEC"], target_list=options.target_list, output=p["TARGET_LIST_FILE"])
    if options.verbose :
        print("Target list table:")
        print(target_list)

    # log messages:
    logger.info("Identifying detector modes")
    # detect all detector modes
    detector_modes = s4db.get_detector_modes_observed(db, science_only=True, detector_keys=p["DETECTOR_MODE_KEYWORDS"])

    # set astrometry ref image as the one for this channel
    p["ASTROM_REF_IMG"] = os.path.join(p["CALIBDB_DIR"], p["ASTROM_REF_IMGS"][j])
    # log messages:
    logger.info("Astrometric reference image: {}".format(p["ASTROM_REF_IMG"]))
    
    # set object id if provided in the command line
    object_id = None
    if options.object != "" :
        object_id = options.object
    
    for key in detector_modes.keys():
           
        # log messages:
        logger.info("Reducing detector mode: {} ".format(key))
    
        # log messages:
        logger.info("Running master zero calibration")
    
        # Run master zero for a given detector mode
        p = s4pipelib.run_master_zero_calibration(p, db, options.nightdir, data_dir, reduce_dir, p['CHANNELS'][j], detector_modes[key], key, force=options.force)
           
        # log messages:
        logger.info("Running master flat calibrations")
    
        # Run master flat for PHOT, POLAR_L2 and POLAR_L4 modes
        p_phot, p_polarl2, p_polarl4 = s4pipelib.run_master_flat_calibrations(p, db, options.nightdir, data_dir, reduce_dir, p['CHANNELS'][j], detector_modes[key], key, force=options.force)
    
        try:
            # log messages:
            logger.info("Start reduction of science data in {} mode".format(p_phot['INSTMODE_PHOTOMETRY_KEYVALUE']))
        
            # reduce science data in photometric mode
            p_phot = s4pipelib.reduce_sci_data(db, p_phot, j, p_phot['INSTMODE_PHOTOMETRY_KEYVALUE'], detector_modes[key], options.nightdir, reduce_dir, polar_mode=None, fit_zero=False, detector_mode_key=key, obj=object_id, calw_modes=["OFF","None"], match_frames=p['MATCH_FRAMES'], force=options.force, plot_stack=options.plot, plot_lc=options.plot, plot_polar=False)
        except Exception as e:
            # log messages:
            logger.warn("Could not reduce {} mode, detector mode {} : {}".format(p_phot['INSTMODE_PHOTOMETRY_KEYVALUE'], key, e))
        
        try:
            # log messages:
            logger.info("Start reduction of science data in {} {} mode".format(p_polarl2['INSTMODE_POLARIMETRY_KEYVALUE'], p_polarl2['POLARIMETRY_L2_KEYVALUE']))

            # reduce science data in polarimetric L2 mode
            p_polarl2 = s4pipelib.reduce_sci_data(db, p_polarl2, j, p_polarl2['INSTMODE_POLARIMETRY_KEYVALUE'], detector_modes[key], options.nightdir, reduce_dir, polar_mode=p_polarl2['POLARIMETRY_L2_KEYVALUE'], fit_zero=False, detector_mode_key=key, obj=object_id, calw_modes=p_polarl2['CALW_MODES'], match_frames=p['MATCH_FRAMES'], force=options.force, plot_stack=options.plot, plot_lc=options.plot, plot_polar=p["PLOT_POLARIMETRY_FIT"])
        except Exception as e:
            # log messages:
            logger.warn("Could not reduce {}-{} mode, detector mode {} : {}".format(p_polarl2['INSTMODE_POLARIMETRY_KEYVALUE'], p_polarl2['POLARIMETRY_L2_KEYVALUE'], key, e))
        
        try:
            # log messages:
            logger.info("Start reduction of science data in {} {} mode".format(p_polarl4['INSTMODE_POLARIMETRY_KEYVALUE'], p_polarl4['POLARIMETRY_L4_KEYVALUE']))

            # reduce science data in  polarimetric L4 mode
            p_polarl4 = s4pipelib.reduce_sci_data(db, p_polarl4, j, p_polarl4['INSTMODE_POLARIMETRY_KEYVALUE'], detector_modes[key], options.nightdir, reduce_dir, polar_mode=p_polarl4['POLARIMETRY_L4_KEYVALUE'], fit_zero=p['FIT_ZERO_OF_WPPOS_IN_L4'], detector_mode_key=key, obj=object_id, calw_modes=p_polarl4['CALW_MODES'], match_frames=p['MATCH_FRAMES'], force=options.force, plot_stack=options.plot, plot_lc=options.plot, plot_polar=p_polarl4["PLOT_POLARIMETRY_FIT"])
        except Exception as e:
            # log messages:
            logger.warn("Could not reduce {}-{} mode, detector mode {} : {}".format(p_polarl4['INSTMODE_POLARIMETRY_KEYVALUE'], p_polarl4['POLARIMETRY_L4_KEYVALUE'], key, e))
        
        # log messages:
        logger.info("Finished reducing channel {} detector mode {} ".format(channel,key))
        
    # log messages:
    logger.info("Closing SPARC4 channel {} ".format(channel))
    # close logger
    logger.handlers.clear()

