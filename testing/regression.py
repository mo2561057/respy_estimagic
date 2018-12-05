#!/usr/bin/env python
"""This module is a first take at regression tests."""
import pickle as pkl
import numpy as np

from respy.tests.codes.random_init import write_init_file
from respy.tests.codes.random_init import generate_init

from respy_smm.tests.auxiliary import get_ingredients
from respy_smm import PACKAGE_DIR
from respy_smm import optimize

np.random.seed(123)

# Generate a new regression vault ...
if False:

    NUM_TESTS = 10

    tests = []
    for _ in range(NUM_TESTS):
        print(_)
        constr = dict()
        constr['flag_interpolation'] = False
        constr['flag_myopic'] = False
        dict_ = generate_init(constr)

        # TODO: This needs to be done for random point of evaluation.
        _, moments_obs, _, weighing_matrix = get_ingredients('test.respy.ini')

        # TODO: This needs to be a general function once more optimization algorithms are available.
        toolbox = 'nag'
        toolbox_spec = dict()
        toolbox_spec['algorithm'] = 'bobyqa'
        toolbox_spec['max_evals'] = np.random.randint(1, 10)

        rslt = optimize('test.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)
        fval = rslt['Step'].iloc[0]

        tests += [[dict_, toolbox, toolbox_spec, fval]]

        pkl.dump(tests, open(PACKAGE_DIR + '/tests/regression_vault.respy_smm.pkl', 'wb'))


tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.respy_smm.pkl', 'rb'))

for test in tests:

    dict_, toolbox, toolbox_spec, fval = test
    write_init_file(dict_)

    _, moments_obs, _, weighing_matrix = get_ingredients('test.respy.ini')

    rslt = optimize('test.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)
    rslt = rslt['Step'].iloc[0]

    np.testing.assert_almost_equal(rslt, fval)
