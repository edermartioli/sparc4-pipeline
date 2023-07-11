# This file is part of the SPARC4 Pipeline distribution
# https://github.com/sparc4-dev/sparc4-pipeline
# Copyright (c) 2023 Eder Martioli and Julio Campagnolo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from copy import deepcopy

import numpy as np
from astropy.io import fits
from astropy.table import Table


def create_db_from_observations(filelist: list,
                                dbkeys: list = None,
                                include_img_statistics: bool = True,
                                include_only_fullframe: bool = True,
                                output: str = "") -> Table:
    """Create a database of observations.

    Parameters
    ----------
    filelist : list
        input file list
    dbkeys : list, optional
        input list of header keywords to include in database
    include_img_statistics : bool, optional
        boolean to include image statistics in the database (slower)
    output : str, optional
        output db FITS file name

    Returns
    -------
    tbl : astropy.table
        data base in astropy table format
    """
    # Avoid security issues with mutable default arguments
    dbkeys = dbkeys or []

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
                    continue
        except Exception:
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

    # initialize dict as data container
    tbl = Table(tbldata)

    if output != "":
        tbl.write(output, overwrite=True)

    return tbl


def create_db_from_file(db_filename: str) -> Table:
    """Create a database from an input db file.

    Parameters
    ----------
    db_filename : str
        input db file

    Returns
    -------
    tbl : astropy.table
        data base in astropy table format
    """
    tbl = Table(fits.getdata(db_filename, 1))
    return tbl


def get_targets_observed(tbl: Table,
                         inst_mode: str = None,
                         polar_mode: str = None,
                         calwheel_mode: str = None,
                         detector_mode: str = None) -> Table:
    """Get targets observed.

    Parameters
    ----------
    tbl : tbl : astropy.table
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

    targets = tbl.group_by("OBJECT").groups.keys

    return targets


def get_detector_modes_observed(tbl: Table,
                                science_only: bool = True,
                                detector_keys: list = None) -> dict:
    """Get detector modes observed.

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

    # detector_keys = ["VBIN", "HBIN", "INITLIN", "INITCOL", "FINALLIN",
    #                  "FINALCOL", "VCLKAMP", "CCDSERN", "VSHIFT", "PREAMP",
    #                  "READRATE", "EMMODE", "EMGAIN"]

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


def get_inst_modes_observed(tbl: Table,
                            science_only: bool = True) -> Table:
    """Get instrument modes observed.

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

    inst_modes = tbl.group_by("INSTMODE").groups.keys

    return inst_modes


def get_polar_modes_observed(tbl: Table,
                             science_only: bool = True) -> Table:
    """Get polarimetry modes observed.

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
        tbl = tbl[np.logical_and(tbl["OBSTYPE"] == "OBJECT",
                                 tbl["INSTMODE"] == "POLAR")]
    else:
        tbl = tbl[tbl["INSTMODE"] == "POLAR"]

    polar_modes = tbl.group_by("WPSEL").groups.keys

    return polar_modes


def get_calib_wheel_modes(tbl: Table,
                          science_only: bool = True,
                          polar_only: bool = True) -> Table:
    """Get calibration wheel modes observed.

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

    calwheel_modes = tbl.group_by("CALW").groups.keys

    return calwheel_modes


def get_file_list(tbl: Table,
                  object_id: str = None,
                  inst_mode: str = None,
                  polar_mode: str = None,
                  obstype: str = None,
                  calwheel_mode: str = None,
                  detector_mode: dict = None,
                  skyflat: bool = False) -> list:
    """Get a list of files selected from database.

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
    skyflat : bool, optional
        special case for sky flats

    Returns
    -------
    outlist : list
        list of selected files from database
    """

    if object_id is not None:
        tbl = tbl[tbl["OBJECT"] == object_id]

    if skyflat:
        tbl = tbl[(tbl["OBSTYPE"] == 'SFLAT') | (tbl["OBSTYPE"] == 'SKYFLAT') |
                  (tbl["OBJECT"] == 'SFLAT') | (tbl["OBJECT"] == 'SKYFLAT')]

    if obstype in ["ZERO", "FLAT", "OBJECT"]:  # already ensured not None
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

    outlist = []
    for i in range(len(tbl)):
        outlist.append(str(tbl["FILE"][i]))

    return outlist


def get_polar_sequences(tbl: Table,
                        object_id: str,
                        detector_mode: dict,
                        polar_mode: str,
                        calwheel_mode: str = None) -> list:
    """Get polar sequences within a given mode.

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
