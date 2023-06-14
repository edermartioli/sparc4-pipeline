import yaml
import os


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
