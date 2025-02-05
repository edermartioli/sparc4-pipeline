"""
    Created on Jun 14 2023

    Description: Library for the SPARC4 pipeline database

    @author: Eder Martioli <martioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

from copy import deepcopy

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii

def create_db_from_observations(filelist, dbkeys=[], include_img_statistics=True, include_only_fullframe=True, valid_obstype_keys=["ZERO", "FLAT", "FOCUS", "DARK", "OBJECT"], mandatory_keys=[], output=""):
    """ SPARC4 pipeline module to create a database of observations
    Parameters
    ----------
    filelist : list
        input file list
    dbkeys : list, optional
        input list of header keywords to include in database
    include_img_statistics : bool, optional
        boolean to include image statistics in the database (slower)
    valid_obstype_keys : list of str
        list of valide obstype key values
    mandatory_keys : list of str
        list of mandatory keywords
    output : str, optional
        output db FITS file name

    Returns
    -------
    tbl : astropy.table
        data base in astropy table format
    """
    tbldata = {}

    tbldata["FILE"] = []

    if include_img_statistics:
        tbldata["MAX"] = []
        tbldata["MIN"] = []
        tbldata["MEAN"] = []
        tbldata["MEDIAN"] = []
        tbldata["STDDEV"] = []

    for key in dbkeys:
        tbldata[key] = []

    for i in range(len(filelist)):

        try:
            # open fits file
            hdul = fits.open(filelist[i])
            hdr = hdul[0].header

            if include_only_fullframe:
                if hdr["NAXIS1"] != 1024 or hdr["NAXIS2"] != 1024:
                    print("Full image is required, skipping image file {}".format(filelist[i]))
                    continue
            if "OBSTYPE" in hdr.keys() :
                if hdr["OBSTYPE"] not in valid_obstype_keys :
                    print("OBSTYPE={} is not in list of valid keys, skipping image file {}".format(hdr["OBSTYPE"],filelist[i]))
                    continue
            else :
                print("Keyword OBSTYPE not found, skipping image file {}".format(filelist[i]))
                continue
                
            for j in range(len(mandatory_keys)) :
                if mandatory_keys[j] not in hdr.keys() :
                    print("Mandatory keyword {} not found, skipping image file {}".format(mandatory_keys[j],filelist[i]))
                    continue
        except:
            print("Skipping image file {}".format(filelist[i]))
            continue

        tbldata["FILE"].append(filelist[i])

        if include_img_statistics:
            tbldata["MAX"].append(np.nanmax(hdul[0].data))
            tbldata["MIN"].append(np.nanmin(hdul[0].data))
            tbldata["MEAN"].append(np.nanmean(hdul[0].data))
            tbldata["MEDIAN"].append(np.nanmedian(hdul[0].data))
            tbldata["STDDEV"].append(np.nanstd(hdul[0].data))
        else:
            hdr = fits.getheader(filelist[i], 0)

        for key in dbkeys:
            tbldata[key].append(hdr[key])
    
    tbl = None
    if len(tbldata["FILE"]) :
        # initialize dict as data container
        tbl = Table(tbldata)

        if output != "":
            tbl.write(output, overwrite=True)

    return tbl


def create_db_from_file(db_filename):
    """ SPARC4 pipeline module to create a database from an input db file
    Parameters
    ----------
    db_filename : str
        input db file

    Returns
    -------
    tbl : astropy.table
        data base in astropy table format
    """
    if db_filename.endswith(".fits") :
        tbl = Table(fits.getdata(db_filename, 1))
    else :
        tbl = ascii.read(db_filename)
    return tbl


def get_targets_observed(intbl, inst_mode=None, polar_mode=None, calwheel_mode=None, detector_mode=None):
    """ SPARC4 pipeline module to get targets observed
    Parameters
    ----------
    intbl : astropy.table
        input database table
    inst_mode : str, optional
        to select observations of a given instrument mode
    polar_mode : str, optional
        to select observations of a given polarimetric mode
    calwheel_mode : str, optional
        to select observations of a given calibration wheel position mode
    detector_mode : dict, optional
        to select observations of a given detector mode

    Returns
    -------
    targets : astropy.table
        objects observed detected in database
    """
    tbl = deepcopy(intbl)
    
    tbl = tbl[tbl["OBSTYPE"] == "OBJECT"]

    if detector_mode is not None:
        for key in detector_mode.keys():
            tbl = tbl[tbl[key] == detector_mode[key]]

    if polar_mode is not None:
        tbl = tbl[tbl['WPSEL'] == polar_mode]

    if inst_mode is not None:
        tbl = tbl[tbl['INSTMODE'] == inst_mode]

    if calwheel_mode is not None:
        tbl = tbl[tbl['CALW'] == calwheel_mode]

    print(tbl)
    targets = []
    if len(tbl) != 0 :
        targets = tbl.group_by("OBJECT").groups.keys
        
    return targets


def get_detector_modes_observed(tbl, science_only=True, detector_keys=None):
    """ SPARC4 pipeline module to get detector modes observed
    Parameters
    ----------
    tbl : tbl : astropy.table
        input database table
    science_only : bool, optional
        to consider only science data in mode selection
    detector_keys : list, optional
        list of keywords to match detector modes

    Returns
    -------
    modes : dict
        dictionary with detector mdes detected in database
    """

    # detector_keys = ["VBIN", "HBIN", "INITLIN", "INITCOL", "FINALLIN", "FINALCOL", "VCLKAMP", "CCDSERN", "VSHIFT", "PREAMP", "READRATE", "EMMODE", "EMGAIN"]

    modes = {}

    if science_only:
        tbl = tbl[tbl["OBSTYPE"] == "OBJECT"]

    for i in range(len(tbl)):
        mode_name, detector_mode = "", {}
        for key in detector_keys:
            mode_name += "_{}".format(str(tbl[key][i]).replace(" ", ""))
            detector_mode[key] = tbl[key][i]
        if mode_name not in modes.keys():
            modes[mode_name] = detector_mode

    return modes


def get_inst_modes_observed(tbl, science_only=True):
    """ SPARC4 pipeline module to get instrument modes observed
    Parameters
    ----------
    tbl : tbl : astropy.table
        input database table
    science_only : bool, optional
        to consider only science data in mode selection
    Returns
    -------
    inst_modes : astropy.table
        instrument modes detected in database
    """

    if science_only:
        tbl = tbl[tbl["OBSTYPE"] == "OBJECT"]

    inst_modes = []
    if len(tbl) != 0 :
        inst_modes = tbl.group_by("INSTMODE").groups.keys
        
    return inst_modes


def get_polar_modes_observed(tbl, science_only=True):
    """ SPARC4 pipeline module to get polarimetry modes observed
    Parameters
    ----------
    tbl : tbl : astropy.table
        input database table
    science_only : bool, optional
        to consider only science data in mode selection
    Returns
    -------
    polar_modes : astropy.table
        polarimetry modes detected in database
    """

    if science_only:
        tbl = tbl[(tbl["OBSTYPE"] == "OBJECT") & (tbl["INSTMODE"] == "POLAR")]
    else:
        tbl = tbl[tbl["INSTMODE"] == "POLAR"]

    polar_modes = []
    if len(tbl) != 0 :
        polar_modes = tbl.group_by("WPSEL").groups.keys

    return polar_modes


def get_calib_wheel_modes(tbl, science_only=True, polar_only=True):
    """ SPARC4 pipeline module to get calibration wheel modes observed
    Parameters
    ----------
    tbl : tbl : astropy.table
        input database table
    science_only : bool, optional
        to consider only science data for mode selection
    polar_only : bool, optional
        to consider only data in polar mode for selection

    Returns
    -------
    calwheel_modes : astropy.table
        calibration wheel modes detected in database
    """

    if science_only:
        tbl = tbl[tbl["OBSTYPE"] == "OBJECT"]

    if polar_only:
        tbl = tbl[tbl["INSTMODE"] == "POLAR"]

    calwheel_modes = []
    if len(tbl) != 0 :
        calwheel_modes = tbl.group_by("CALW").groups.keys
            
    return calwheel_modes


def get_file_list(tbl, object_id=None, inst_mode=None, polar_mode=None, obstype=None, calwheel_mode=None, detector_mode=None, wppos=None, skyflat=False):
    """ SPARC4 pipeline module to get a list of files selected from database
    Parameters
    ----------
    tbl : tbl : astropy.table
        input database table
    object_id : str, optional
        to select observations of a given object_id
    inst_mode : str, optional
        to select observations of a given instrument mode
    polar_mode : str, optional
        to select observations of a given polarimetric mode
    obstype : str, optional
        to select observations of a given type (ZERO, FLAT or OBJECT)
    calwheel_mode : str, optional
        to select observations of a given calibration wheel position mode
    detector_mode : dict, optional
        to select observations of a given detector mode
    wppos : int, optional
        to select observations of a given waveplate position
    skyflat : bool, optional
        special case for sky flats
    Returns
    -------
    outlist : list
        list of selected files from database
    """

    # tbl, inst_mode="PHOT", polar_mode="NONE", obstype="ZERO", detector_mode={"PREAMP": "Gain 2","READRATE": 1,"EMMODE": 'Conventional',"EMGAIN": 2}

    if object_id is not None:
        tbl = tbl[tbl["OBJECT"] == object_id]

    if skyflat:
        tbl = tbl[(tbl["OBSTYPE"] == 'SFLAT') | (tbl["OBSTYPE"] == 'SKYFLAT') | (
            tbl["OBJECT"] == 'SFLAT') | (tbl["OBJECT"] == 'SKYFLAT')]

    if (obstype is not None) and (obstype in ["ZERO", "FLAT", "OBJECT"]):
        tbl = tbl[tbl["OBSTYPE"] == obstype]

    if detector_mode is not None:
        for key in detector_mode.keys():
            tbl = tbl[tbl[key] == detector_mode[key]]

    if polar_mode is not None:
        tbl = tbl[tbl['WPSEL'] == polar_mode]

    if inst_mode is not None:
        tbl = tbl[tbl['INSTMODE'] == inst_mode]

    if calwheel_mode is not None:
        tbl = tbl[tbl['CALW'] == calwheel_mode]

    if wppos is not None:
        tbl = tbl[tbl["WPPOS"] == wppos]

    outlist = []
    if len(tbl) == len(tbl.columns) and len(tbl["FILE"][0]) == 1:
        outlist.append(str(tbl["FILE"]))
    else :
        for i in range(len(tbl)):
            outlist.append(str(tbl["FILE"][i]))

    return outlist


def get_polar_sequences(tbl, object_id, detector_mode, polar_mode, calwheel_mode=None):
    """ SPARC4 pipeline module to get polar sequences within a given mode
    Parameters
    ----------
    tbl : tbl : astropy.table
        input database table
    object_id : str
        to select observations of a given object ID
    detector_mode : {}
        to select observations of a given detector mode
    polar_mode : str
        to select observations of a given polarimetric mode
    calwheel_mode : str, optional
        to select observations of a given calibration wheel position mode


    Returns
    -------
    outlists : list of lists
        lists of sequences of files
    """

    tbl = tbl[tbl['INSTMODE'] == 'POLAR']
    tbl = tbl[tbl["OBJECT"] == object_id]
    tbl = tbl[tbl["OBSTYPE"] == "OBJECT"]
    for key in detector_mode.keys():
        tbl = tbl[tbl[key] == detector_mode[key]]
    tbl = tbl[tbl['WPSEL'] == polar_mode]

    if calwheel_mode != None:
        tbl = tbl[tbl['CALW'] == calwheel_mode]

    tbl.sort("DATE-OBS")

    sequences = []
    # print("******** START NEW SEQUENCE *********")

    if len(tbl["WPPOS"]):
        prev_pos = deepcopy(tbl["WPPOS"][0])
        seq = []

        for i in range(len(tbl)):
            current_pos = tbl["WPPOS"][i]

            # print(i, tbl["FILE"][i], tbl["WPPOS"][i], current_pos, prev_pos)

            if current_pos < prev_pos:
                sequences.append(seq)
                seq = []
                # print("******** START NEW SEQUENCE *********")

            seq.append(tbl["FILE"][i])
            prev_pos = current_pos

            if i == len(tbl) - 1:
                sequences.append(seq)

    return sequences

