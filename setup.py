#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from setuptools import setup

VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
except Exception:
    version = '{version}'
""".lstrip()

setup(use_scm_version={'write_to': os.path.join('sparc4', 'version.py'),
                       'write_to_template': VERSION_TEMPLATE})
