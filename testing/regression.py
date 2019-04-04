#!/usr/bin/env python
"""This module is a first take at regression tests."""
import pickle as pkl
import argparse
import os

from ose_utils.testing import create_regression_vault, check_regression_vault
from respy_smm.tests.auxiliary import run_regression_test
from respy_smm import PACKAGE_DIR

# The regression tests create too many parallel processes at this point if run in parallel.
assert 'PMI_SIZE' not in os.environ.keys()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run regression tests')

    parser.add_argument('-t', '--tests', type=int, help='number of tests to analyze',
                        default=1, dest='num_tests', required=True)

    parser.add_argument('--create', action='store_true', dest='is_create',
                        help='create vault (instead of checking)')

    args = parser.parse_args()

    if args.is_create:
        vault = create_regression_vault(run_regression_test, args.num_tests)
        pkl.dump(vault, open(PACKAGE_DIR + '/tests/regression_vault.pkl', 'wb'))

    vault = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.pkl', 'rb'))
    check_regression_vault(run_regression_test, args.num_tests, vault)
