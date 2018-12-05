"""This module contains some basic configuration for the package."""
import warnings
import os

from respy_smm.auxiliary import warning_on_one_line
import respy_smm


HUGE_INT = 1000000000
HUGE_FLOAT = 1e15

DEFAULT_BOUND = 1e8

PACKAGE_DIR = os.path.abspath(respy_smm.__file__).replace('__init__.py', '')
IS_DEBUG = 'IS_DEVELOPMENT' in os.environ.keys()

# We need to manage our warnings
warnings.formatwarning = warning_on_one_line
if not IS_DEBUG:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
