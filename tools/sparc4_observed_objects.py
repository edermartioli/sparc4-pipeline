"""
    Created on Jun 22 2023

    Description: Module to check all observed objects in a set of SPARC4 data

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI

    Simple usage example:

    python sparc4_distribute_data.py -1

    python sparc4_observed_objects.py --nightdir=20230604 --datadir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_jun23/ --reducedir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_jun23/reduced --wildcard="*.fits"

    """

import glob
import sparc4.db as s4db
# import sparc4.utils as s4utils
import sparc4.pipeline_lib as s4pipelib
# import sparc4.product_plots as s4plt
import os
import sys
from optparse import OptionParser
sys.path.append(os.path.dirname(os.getcwd()))


sparc4_pipeline_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-w", "--wildcard", dest="wildcard",
                  help="wild card to select data", type='string', default="")
parser.add_option("-d", "--datadir", dest="datadir",
                  help="data directory", type='string', default="")
parser.add_option("-r", "--reducedir", dest="reducedir",
                  help="Reduced data directory", type='string', default="")
parser.add_option("-c", "--channels", dest="channels",
                  help="SPARC4 channels: e.g '1,3,4' ", type='string', default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir",
                  help="Name of night directory common to all channels", type='string', default="")
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


# loop over selected channels
for channel in p['SELECTED_CHANNELS']:
    # set zero based index of current channel
    j = channel - 1
    data_dir = p['data_directories'][j]
    print("Directory: {}".format(data_dir))

    filelist = glob.glob("{}/{}".format(data_dir, options.wildcard))

    # if db doesn't exist create one
    dbfile = p['s4db_files'][j].replace(".fits", "_tmp.fits")
    db = s4db.create_db_from_observations(
        filelist, p['DB_KEYS'], include_img_statistics=p["INCLUDE_IMG_STATISTICS"], include_only_fullframe=p["FULL_FRAMES_ONLY"], output=dbfile)

    # get list of objects observed in photometric mode
    objs = s4db.get_targets_observed(db)

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
