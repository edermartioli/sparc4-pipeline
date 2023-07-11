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


import os

import yaml


def load_sparc4_parameters(config_file=None):
    """Load SPARC4 pipeline parameters from a yaml file.

    Parameters
    ----------
    config_file : str (optional)
        Path to yaml file containing pipeline parameters. If None, the
        default parameters are loaded.

    Returns
    -------
    params : dict
        Dictionary containing pipeline parameters.
    """
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__),
                                   'sparc4_params.yaml')
    with open(config_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params
