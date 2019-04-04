#!/usr/bin/env python
"""This script runs the regression tests."""
import argparse

from ose_utils.testing import run_property_tests
import respy_smm.tests

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run property tests')

    parser.add_argument('-m', '--minutes', type=float, help='minutes', default=1)

    args = parser.parse_args()

    run_property_tests(respy_smm.tests, args.minutes)


