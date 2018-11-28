"""This module contains some basic configuration for the package."""
import warnings

from respy_smm.auxiliary import warning_on_one_line

warnings.formatwarning = warning_on_one_line

HUGE_INT = 1000000000
HUGE_FLOAT = 1e15

DEFAULT_BOUND = 1e8
