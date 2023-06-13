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

    python sparc4_mini_pipeline.py --nightdir=20230503 -vp
    
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

import numpy as np

sparc4_pipeline_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir", help="data directory",type='string',default="")
parser.add_option("-r", "--reducedir", dest="reducedir", help="Reduced data directory",type='string',default="")
parser.add_option("-c", "--channels", dest="channels", help="SPARC4 channels: e.g '1,3,4' ",type='string',default="1,2,3,4")
parser.add_option("-a", "--nightdir", dest="nightdir", help="Name of night directory common to all channels",type='string',default="")
parser.add_option("-f", action="store_true", dest="force", help="Force reduction", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_mini_pipeline.py")
    sys.exit(1)

match_frames = True
fit_zero_of_wppos = False

# initialize pipeline parameters 
p = s4pipelib.init_s4_p(options.datadir,
                        options.reducedir,
                        options.nightdir,
                        options.channels,
                        print_report=options.verbose)

# Run full reduction for selected channels
for channel in p['SELECTED_CHANNELS'] :
#for j in range(1,2) :
    
    # set zero based index of current channel
    j = channel - 1

    data_dir = p['data_directories'][j]
    ch_reduce_dir = p['ch_reduce_directories'][j]
    reduce_dir = p['reduce_directories'][j]
    
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
    
    for obj in p['objsInPhot'][j] :
        # set suffix for output stack filename
        stack_suffix = "{}_s4c{}_{}".format(options.nightdir,p['CHANNELS'][j],obj.replace(" ",""))

        # get full list of science objects in the current channel
        sci_list = p['objsInPhotdata'][j][obj]

        # run stack and reduce individual science images (produce *_proc.fits)
        p = s4pipelib.stack_and_reduce_sci_images(p,
                                        sci_list,
                                        reduce_dir,
                                        ref_img="",
                                        stack_suffix=stack_suffix,
                                        force=options.force,
                                        match_frames=match_frames,
                                        polarimetry=False,
                                        verbose=options.verbose,
                                        plot=options.plot)

        # set suffix for output time series filename
        ts_suffix = "{}_s4c{}_{}".format(options.nightdir,p['CHANNELS'][j],obj.replace(" ",""))
        # run photometric time series
        phot_ts_product = s4pipelib.phot_time_series(p['OBJECT_REDUCED_IMAGES'][1:],
                                                     ts_suffix=ts_suffix,
                                                     reduce_dir=reduce_dir,
                                                     time_key=p['TIME_KEYWORD_IN_PROC'],
                                                     time_format=p['TIME_FORMAT_IN_PROC'],
                                                     ref_catalog_name=p['PHOT_REF_CATALOG_NAME'],
                                                     catalog_names = p['PHOT_CATALOG_NAMES_TO_INCLUDE'],
                                                     force=options.force)
                    
        if options.plot :
            target = 0
            comps = [1,2,3,4]
            #plot light curve
            s4plt.plot_light_curve(phot_ts_product,
                                   target=target,
                                   comps=comps,
                                   nsig=10,
                                   plot_coords=True,
                                   plot_rawmags=True,
                                   plot_sum=True,
                                   plot_comps=True,
                                   catalog_name=p['PHOT_REF_CATALOG_NAME'])

    for obj in p['objsInPolarl2'][j] :
        # set stack suffix
        stack_suffix = "{}_s4c{}_{}_POL_L2".format(options.nightdir,p['CHANNELS'][j],obj.replace(" ",""))

        # get full list of science objects in the current channel
        sci_list = p['objsInPolarL2data'][j][obj]

        # run stack and reduce individual science images (produce *_proc.fits)
        p = s4pipelib.stack_and_reduce_sci_images(p,
                                        sci_list,
                                        reduce_dir,
                                        ref_img="",
                                        stack_suffix=stack_suffix,
                                        force=options.force,
                                        match_frames=match_frames,
                                        polarimetry=True,
                                        verbose=options.verbose,
                                        plot=options.plot)

        l2_sequences, l2_wppositions = s4utils.select_polar_sequences(p['OBJECT_REDUCED_IMAGES'],
                                                                        max_index_gap=p['MAX_INDEX_GAP_TO_BREAK_POL_SEQS'],
                                                                        max_time_gap=p['MAX_TIME_GAP_TO_BREAK_POL_SEQS'])

        p['PolarL2products'] = []
        for i in range(len(l2_sequences)) :

            print("Running half-wave polarimetry for sequence: {} of {}".format(i+1,len(l2_sequences)))
            polarL2product = s4pipelib.compute_polarimetry(l2_sequences[i],
                                                           wave_plate='halfwave',
                                                           base_aperture=p['APERTURE_INDEX_FOR_PHOTOMETRY_IN_POLAR'],
                                                           compute_k=True,
                                                           zero=0)

            pol_results = s4pipelib.get_polarimetry_results(polarL2product,
                                                            source_index=0,
                                                            min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                                            max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                                            plot=options.plot,
                                                            verbose=options.verbose)
            p['PolarL2products'].append(polarL2product)
        
        
        # set suffix for output time series filename
        ts_suffix = "{}_s4c{}_{}_L2".format(options.nightdir,p['CHANNELS'][j],obj.replace(" ",""))

        # create polar time series product
        polar_ts_L2product = s4pipelib.polar_time_series(p['PolarL2products'],
                                               reduce_dir=reduce_dir,
                                               ts_suffix=ts_suffix,
                                               aperture_index=p['APERTURE_INDEX_FOR_PHOTOMETRY_IN_POLAR'],
                                               min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                               max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                               force=options.force)


    for obj in p['objsInPolarl4'][j] :
        # set stack suffix
        stack_suffix = "{}_s4c{}_{}_POL_L4".format(options.nightdir,p['CHANNELS'][j],obj.replace(" ",""))

        # get full list of science objects in the current channel
        sci_list = p['objsInPolarL4data'][j][obj]

        # run stack and reduce individual science images (produce *_proc.fits)
        p = s4pipelib.stack_and_reduce_sci_images(p,
                                        sci_list,
                                        reduce_dir,
                                        ref_img="",
                                        stack_suffix=stack_suffix,
                                        force=options.force,
                                        match_frames=match_frames,
                                        polarimetry=True,
                                        verbose=options.verbose,
                                        plot=options.plot)

        l4_sequences, l4_wppositions = s4utils.select_polar_sequences(p['OBJECT_REDUCED_IMAGES'],
                                                                        max_index_gap=p['MAX_INDEX_GAP_TO_BREAK_POL_SEQS'],
                                                                        max_time_gap=p['MAX_TIME_GAP_TO_BREAK_POL_SEQS'])

        p['PolarL4products'] = []
        for i in range(len(l4_sequences)) :
            print("Running quarter-wave polarimetry for sequence: {} of {}".format(i+1,len(l4_sequences)))
            polarL4product = s4pipelib.compute_polarimetry(l4_sequences[i],
                                                                wave_plate='quarterwave',
                                                                base_aperture=p['APERTURE_INDEX_FOR_PHOTOMETRY_IN_POLAR'],
                                                                compute_k=False,
                                                                fit_zero=fit_zero_of_wppos,
                                                                zero=p['ZERO_OF_WAVEPLATE'])

            pol_results = s4pipelib.get_polarimetry_results(polarL4product,
                                                            source_index=0,
                                                            min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                                            max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                                            plot=options.plot,
                                                            verbose=options.verbose)
            p['PolarL4products'].append(polarL4product)
    
        # set suffix for output time series filename
        ts_suffix = "{}_s4c{}_{}_L4".format(options.nightdir,p['CHANNELS'][j],obj.replace(" ",""))

        # create polar time series product
        polar_ts_L4product = s4pipelib.polar_time_series(p['PolarL4products'],
                                               reduce_dir=reduce_dir,
                                               ts_suffix=ts_suffix,
                                               aperture_index=p['APERTURE_INDEX_FOR_PHOTOMETRY_IN_POLAR'],
                                               min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                               max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                               force=options.force)
