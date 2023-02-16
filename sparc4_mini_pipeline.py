# -*- coding: iso-8859-1 -*-
"""
    Created on Jun 4 2022
    
    Description: Mini pipeline to process sparc4 data in photometric mode
    
    @author: Eder Martioli <martioli@lna.br>
    
    Laboratório Nacional de Astrofísica - LNA/MCTI
    
    Simple usage example:

    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_mini_pipeline.py --datadir=/Volumes/Samsung_T5/Data/OPD/HATS-20/test/ --reducedir=/Volumes/Samsung_T5/Data/OPD/HATS-20/reduced/

    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_mini_pipeline.py --datadir=/Volumes/Samsung_T5/Data/OPD/HATS-20/fixed/ --reducedir=/Volumes/Samsung_T5/Data/OPD/HATS-20/reduced/

    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_mini_pipeline.py --datadir=/Volumes/Samsung_T5/sparc4-pipeline/teste/ --reducedir=/Volumes/Samsung_T5/sparc4-pipeline/reduced/ -pv


    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_mini_pipeline.py --datadir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/ --reducedir=/Volumes/Samsung_T5/Data/SPARC4/comissioning_nov22/reduced --nightdir=20221104
    
    python /Volumes/Samsung_T5/sparc4-pipeline/sparc4_mini_pipeline.py --nightdir=20221104
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
from optparse import OptionParser

import sparc4_product_plots as s4plt
import sparc4_pipeline_lib as s4pipelib
import sparc4_utils as s4utils
import sparc4_params

sparc4_pipeline_dir = os.path.dirname(__file__)


parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir", help="data directory",type='string',default="")
parser.add_option("-r", "--reducedir", dest="reducedir", help="Reduced data directory",type='string',default="")
parser.add_option("-c", "--channels", dest="channels", help="SPARC4 channels: e.g '1,2,3,4' ",type='string',default="")
parser.add_option("-a", "--nightdir", dest="nightdir", help="Name of night directory common to all channels",type='string',default="")
parser.add_option("-f", action="store_true", dest="force", help="Force reduction", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_mini_pipeline.py")
    sys.exit(1)

polarimetry = False
match_frames = True


# load pipeline parameters
p = sparc4_params.load_sparc4_parameters()

if options.datadir != "" :
    p['ROOTDATADIR'] = options.datadir
    
if options.reducedir != "" :
    p['ROOTREDUCEDIR'] = options.reducedir
    
if options.channels != "" :
    p['CHANNELS']  = []
    channels = options.channels.split(",")
    for ch in channels :
        p['CHANNELS'].append(int(ch))
        
# if reduced dir doesn't exist create one
if not os.path.exists(p['ROOTREDUCEDIR']) :
    os.mkdir(p['ROOTREDUCEDIR'])


#organize files to be reduced
p = s4utils.identify_files(p, options.nightdir, print_report=True)


for j in range(len(p['CHANNELS'])) :
    
    data_dir = p['data_directories'][j]
    
    ch_reduce_dir = '{}/sparc4acs{}/'.format(p['ROOTREDUCEDIR'],p['CHANNELS'][j])
    reduce_dir = '{}/sparc4acs{}/{}/'.format(p['ROOTREDUCEDIR'],p['CHANNELS'][j],options.nightdir)

    if not os.path.exists(ch_reduce_dir) :
        os.mkdir(ch_reduce_dir)

    # if reduced dir doesn't exist create one
    if not os.path.exists(reduce_dir) :
        os.mkdir(reduce_dir)

    # calculate master bias
    master_zero = "{}/{}_s4c{}_MasterZero.fits".format(reduce_dir,options.nightdir,p['CHANNELS'][j])
    p = s4pipelib.run_master_calibration(p, inputlist=p['zeros'][j], output=master_zero, obstype='bias', data_dir=data_dir, reduce_dir=reduce_dir, force=options.force)

    if len(p['sflats'][j]) :
        # calculate master sky flat
        master_skyflat = "{}/{}_s4c{}_MasterSkyFlat.fits".format(reduce_dir,options.nightdir,p['CHANNELS'][j])
        p = s4pipelib.run_master_calibration(p, inputlist=p['sflats'][j], output=master_skyflat, obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=options.force)

    # calculate master dome flat
    master_dflat = "{}/{}_s4c{}_MasterDomeFlat.fits".format(reduce_dir,options.nightdir,p['CHANNELS'][j])
    p = s4pipelib.run_master_calibration(p, inputlist=p['dflats'][j], output=master_dflat, obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=options.force)

    p["ASTROM_REF_IMG"] = p["ASTROM_REF_IMGS"][j]
    
    """
    for object in p['objsInPhot'][j] :
        stack_suffix = "{}_s4c{}_{}".format(options.nightdir,p['CHANNELS'][j],object.replace(" ",""))
        # reduce science data and calculate stack
        p = s4pipelib.reduce_science_images(p, p['objsInPhotdata'][j][object], data_dir=data_dir, reduce_dir=reduce_dir, force=options.force, match_frames=match_frames, stack_suffix=stack_suffix, polarimetry=False)
        if match_frames and options.plot :
            s4plt.plot_sci_frame(p['OBJECT_STACK'], nstars=20)
    """
    for object in p['objsInPolarl2'][j] :
        if object == "NGC20241" :
            stack_suffix = "{}_s4c{}_{}_POL_L2".format(options.nightdir,p['CHANNELS'][j],object.replace(" ",""))
            # reduce science data and calculate stack
            p = s4pipelib.reduce_science_images(p, p['objsInPolarL2data'][j][object], data_dir=data_dir, reduce_dir=reduce_dir, force=options.force, match_frames=match_frames, stack_suffix=stack_suffix, polarimetry=True)
            
            if match_frames and options.plot :
                s4plt.plot_sci_polar_frame(p['OBJECT_STACK'])

            l2_sequences ,l2_wppositions = s4utils.select_polar_sequences(p['OBJECT_REDUCED_IMAGES'], max_index_gap=1, max_time_gap=0.04166)

            #p['PolarL2sequences'][object] = l2_sequences

            for i in range(len(l2_sequences)) :
                print("Running half-wave polarimetry for sequence: {} of {}".format(i+1,len(l2_sequences)))
                p['PolarL2products'] = s4pipelib.compute_polarimetry(sci_list, wave_plate='half_wave', compute_k=True, zero=0)

    """
    for object in p['objsInPolarl4'][j] :
        stack_suffix = "{}_s4c{}_{}_POL_L4".format(options.nightdir,p['CHANNELS'][j],object.replace(" ",""))
        # reduce science data and calculate stack
        p = s4pipelib.reduce_science_images(p, p['objsInPolarL4data'][j][object], data_dir=data_dir, reduce_dir=reduce_dir, force=options.force, match_frames=match_frames, stack_suffix=stack_suffix, polarimetry=True)
        if match_frames and options.plot :
            s4plt.plot_sci_polar_frame(p['OBJECT_STACK'])
    """


"""
# calculate master bias
p = s4pipelib.run_master_calibration(p, obstype='bias', data_dir=data_dir, reduce_dir=reduce_dir, force=options.force)

# calculate master flat
p = s4pipelib.run_master_calibration(p, obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=options.force)

# reduce science data and calculate stack
p = s4pipelib.reduce_science_images(p, data_dir=data_dir, reduce_dir=reduce_dir, force=options.force, match_frames=match_frames, polarimetry=polarimetry)

if match_frames and options.plot :
    # plot stack image
    if polarimetry :
        s4plt.plot_sci_polar_frame(p['OBJECT_STACK'])
    else :
        s4plt.plot_sci_frame(p['OBJECT_STACK'], nstars=20)

exit()

target = 0
comps = [1,2,3,4,5,6,7,8,9,10]

# run photometric time series
phot_ts_product = s4pipelib.phot_time_series(p['OBJECT_REDUCED_IMAGES'], target=target, comps=comps, reduce_dir=reduce_dir)

#nstars = 7
# plot light curve
s4plt.plot_light_curve(phot_ts_product, target=target, comps=comps)

"""
