"""
    Created on Jun 21 2023

    Description: Module to distribute SPARC4 data to a target directory

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:


    python sparc4_distribute_data.py -1

    python sparc4_distribute_data.py --nightdir=20230604 --datadir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_jun23/ --reducedir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_jun23/reduced --destdir=/Volumes/Samsung_T5/Data/SPARC4/distribution/ --objects="'H 1722+119','PG 1553+113','PKS 0118-272','PKS 1510-089'"

    python sparc4_distribute_data.py --nightdir=20230604 --datadir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_jun23/ --reducedir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_jun23/reduced --destdir=/Volumes/Samsung_T5/Data/SPARC4/distribution/ --objects="H 1722+119,PG 1553+113,PKS 0118-272,PKS 1510-089" -k -b

    """

import sparc4.db as s4db
# import sparc4.utils as s4utils
import sparc4.pipeline_lib as s4pipelib
# import sparc4.product_plots as s4plt
import os
import sys
from optparse import OptionParser
sys.path.append(os.path.dirname(os.getcwd()))


def transfer_files(inputlist, dest_dir, symboliclinks=False, doit=False, force=False, printcommand=True, create_dest_dir=True):
    cpstr = "cp"
    if symboliclinks:
        cpstr = "ln -s"

    if create_dest_dir and not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for i in range(len(inputlist)):
        basename = os.path.basename(inputlist[i])
        output = os.path.join(dest_dir, basename)
        command = "{} {} {}".format(cpstr, inputlist[i], output)
        if printcommand:
            print(command)
        if not os.path.exists(output) or force:
            if doit:
                os.system(command)
        else:
            print("Skipping existing file: {}".format(inputlist[i]))


sparc4_pipeline_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir",
                  help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",
                  help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",
                  help="SPARC4 channels: e.g '1,3,4' ", type='string', default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",
                  help="Name of night directory common to all channels", type='string', default="")
parser.add_option("-o", "--objects", dest="objects",
                  help="List of objects to get data", type='string', default="")
parser.add_option("-e", "--destdir", dest="destdir",
                  help="Destination data directory", type='string', default="")
parser.add_option("-1", action="store_true", dest="checkdata",
                  help="checkdata", default=False)
parser.add_option("-k", action="store_true",
                  dest="symboliclinks", help="symboliclinks", default=False)
parser.add_option("-b", action="store_true",
                  dest="onlycalibration", help="onlycalibration", default=False)
parser.add_option("-p", action="store_true", dest="plot",
                  help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose",
                  help="verbose", default=False)

try:
    options, args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_distribute_data.py")
    sys.exit(1)

doit = True

# initialize pipeline parameters
p = s4pipelib.init_s4_p(options.nightdir,
                        options.datadir,
                        options.reducedir,
                        options.channels,
                        print_report=False)


if options.checkdata:
    # loop over selected channels
    for channel in p['SELECTED_CHANNELS']:
        # set zero based index of current channel
        j = channel - 1
        data_dir = p['data_directories'][j]
        print("Directory: {}".format(data_dir))

        # if db doesn't exist create one
        if not os.path.exists(p['s4db_files'][j]):
            db = s4db.create_db_from_observations(p['filelists'][j], p['DB_KEYS'], include_img_statistics=p["INCLUDE_IMG_STATISTICS"],
                                                  include_only_fullframe=p["FULL_FRAMES_ONLY"], output=p['s4db_files'][j])
        else:
            db = s4db.create_db_from_file(p['s4db_files'][j])

        # get list of objects observed in photometric mode
        objs = s4db.get_targets_observed(db)

        # loop over each object
        for k in range(len(objs)):
            obj = objs[k][0]

            objdb = db[db["OBJECT"] == obj]

            print("Object: {}".format(obj))
            # detect all detector modes
            detector_modes = s4db.get_detector_modes_observed(
                objdb, science_only=True, detector_keys=p["DETECTOR_MODE_KEYWORDS"])
            instmodes = s4db.get_inst_modes_observed(objdb, science_only=True)

            for key in detector_modes.keys():
                print("Detector mode: {}".format(key))
            for i in range(len(instmodes)):
                print("Inst mode: {}".format(instmodes[i][0]))
            if instmodes[i][0] == "POLAR":
                polarmodes = s4db.get_polar_modes_observed(
                    objdb, science_only=True)
                for i in range(len(polarmodes)):
                    print("Polar mode: {}".format(polarmodes[i][0]))
            print("---------------------\n")

        listofobjects = []
        strlist = '"'
        for k in range(len(objs)):
            obj = objs[k][0]
            listofobjects.append(obj)
            if k == len(objs) - 1:
                strlist += '{}"'.format(obj)
            else:
                strlist += '{},'.format(obj)
        print(strlist)
    exit()


in_objs = options.objects.split(",")

if in_objs == ['']:
    print("ERROR: Empty list of objects. Provide an object list with the option --objects='obj1,obj2,obj3'")
    exit()

# loop over selected channels
for channel in p['SELECTED_CHANNELS']:
    # set zero based index of current channel
    j = channel - 1
    data_dir = p['data_directories'][j]
    dest_dir = '{}/sparc4acs{}/{}/'.format(options.destdir,
                                           p['CHANNELS'][j], options.nightdir)

    # if db doesn't exist create one
    if not os.path.exists(p['s4db_files'][j]):
        db = s4db.create_db_from_observations(p['filelists'][j], p['DB_KEYS'], include_img_statistics=p["INCLUDE_IMG_STATISTICS"],
                                              include_only_fullframe=p["FULL_FRAMES_ONLY"], output=p['s4db_files'][j])
    else:
        db = s4db.create_db_from_file(p['s4db_files'][j])

    # loop over each object
    for k in range(len(in_objs)):
        obj = in_objs[k]

        print("Object: {}".format(obj))
        # detect all detector and instrument modes
        detector_modes = s4db.get_detector_modes_observed(
            db, science_only=True, detector_keys=p["DETECTOR_MODE_KEYWORDS"])

        # loop over each detector mode observed
        for key in detector_modes.keys():
            print("Detector mode: {}".format(key))

            biaslist = s4db.get_file_list(
                db, obstype=p['BIAS_OBSTYPE_KEYVALUE'], detector_mode=detector_modes[key])

            transfer_files(biaslist, dest_dir, symboliclinks=options.symboliclinks,
                           doit=doit, force=False, printcommand=True)

            flatlist = s4db.get_file_list(
                db, obstype=p['FLAT_OBSTYPE_KEYVALUE'], detector_mode=detector_modes[key])

            transfer_files(flatlist, dest_dir, symboliclinks=options.symboliclinks,
                           doit=doit, force=False, printcommand=True)

            if options.onlycalibration:
                continue

            instmodes = s4db.get_inst_modes_observed(db, science_only=True)
            # switch between phot and polar mode
            for instmode in ["PHOT", "POLAR"]:
                print("Instrument mode: {}".format(instmode))
                # continue only if there are data for this inst. mode
                if instmodes[instmodes["INSTMODE"] == instmode]:
                    # If instrument mode is Polar, then check waveplate modes
                    if instmode == "POLAR":
                        polarmodes = s4db.get_polar_modes_observed(
                            db, science_only=True)
                        # switch between waveplate modes
                        for polarmode in ["L2", "L4"]:
                            print("Polar mode: {}".format(polarmode))
                            if polarmodes[polarmodes["WPSEL"] == polarmode]:
                                print("==>>\t", obj, instmode,
                                      polarmode, detector_modes[key])
                                objlist = s4db.get_file_list(db, object_id=obj, inst_mode=instmode, polar_mode=polarmode,
                                                             obstype=p['OBJECT_OBSTYPE_KEYVALUE'], calwheel_mode=None, detector_mode=detector_modes[key])
                                # get list of objects observed
                                if len(objlist):
                                    transfer_files(
                                        objlist, dest_dir, symboliclinks=options.symboliclinks, doit=doit, force=False, printcommand=True)
