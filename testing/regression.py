#!/usr/bin/env python
"""This module is a first take at regression tests."""
import copy
import os

import pickle as pkl
import numpy as np

from respy_smm.auxiliary_depreciation import respy_spec_old_to_new
from respy.tests.codes.random_init import write_init_file
from respy.tests.codes.random_init import generate_init
from respy_smm.tests.auxiliary import get_ingredients
from respy_smm import PACKAGE_DIR
from respy_smm import optimize
import respy

np.random.seed(123)

# Generate a new regression vault ...
if False:

    NUM_TESTS = 100

    tests = []
    for _ in range(NUM_TESTS):
        print(_)
        constr = dict()

        # This requires some additional work to align the disturbances across the MPI machines.
        constr['flag_parallelism_mpi'] = False
        constr['flag_interpolation'] = False
        constr['flag_myopic'] = False
        dict_ = generate_init(constr)

        # TODO: This needs to be done for random point of evaluation.
        _, moments_obs, _, weighing_matrix = get_ingredients('test.respy.ini')

        # TODO" This is a temporary fix where we move the old spec to the new one. We actually
        #  need to make sure that the stored initialization dict is of the old structure. That is
        #  why we work with a copy here.
        dict_copy = copy.deepcopy(dict_)
        optim_paras = respy.RespyCls('test.respy.ini').get_attr('optim_paras')
        dict_copy['SHOCKS']['coeffs'] = respy_spec_old_to_new(optim_paras)[43:53]
        write_init_file(dict_copy)

        # TODO: This needs to be a general function once more optimization algorithms are available.
        toolbox = 'nag'
        toolbox_spec = dict()
        toolbox_spec['algorithm'] = 'bobyqa'

        # In principle I would want multiple evaluations. However, starting the NAG optimizers
        # without bounds everywhere does result in extreme evaluations and the RESPY code is not
        # properly handling those. Maybe it should, but this is for another PR.
        toolbox_spec['max_evals'] = 1

        rslt = optimize('test.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)
        fval = rslt['Step'].iloc[0]

        tests += [[dict_, toolbox, toolbox_spec, fval]]

        os.system('git clean -df')

    pkl.dump(tests, open(PACKAGE_DIR + '/tests/regression_vault.pkl', 'wb'))

if True:
    tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.pkl', 'rb'))

    for test in tests:

        dict_, toolbox, toolbox_spec, fval = test
        write_init_file(dict_)

        _, moments_obs, _, weighing_matrix = get_ingredients('test.respy.ini')

        # TODO" This is a temporary fix where we move the old spec to the new one.
        dict_copy = copy.deepcopy(dict_)
        optim_paras = respy.RespyCls('test.respy.ini').get_attr('optim_paras')
        dict_['SHOCKS']['coeffs'] = respy_spec_old_to_new(optim_paras)[43:53]
        write_init_file(dict_)

        rslt = optimize('test.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)
        rslt = rslt['Step'].iloc[0]

        np.testing.assert_almost_equal(rslt, fval)

        os.system('git clean -df')
