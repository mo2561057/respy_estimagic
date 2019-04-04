#!/usr/bin/env python
"""This module allows for some more elaborate testing before the a pull request."""
import subprocess

minutes_property_tests = 1
num_regression_tests = 10

print(' \n ... running {:} robustness tests'.format(num_regression_tests))
cmd = './regression.py -t {:}'.format(num_regression_tests)
subprocess.check_call(cmd, shell=True)

print(' \n ... running property tests for {:} minutes'.format(minutes_property_tests))
cmd = './property.py -m {:}'.format(minutes_property_tests)
subprocess.check_call(cmd, shell=True)

print('\n')
