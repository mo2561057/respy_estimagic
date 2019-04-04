#!/usr/bin/env python
"""This module contains some tests for the MLE procedures."""
import numpy as np
from respy_smm.MaximumLikelihoodEstimation import MaximumLikelihoodEstimationCls
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy_smm.auxiliary_depreciation import respy_obj_from_new_init
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.tests.codes.auxiliary import write_lagged_start
from respy.tests.codes.auxiliary import write_edu_start
from respy.tests.codes.auxiliary import write_draws
from respy.tests.codes.auxiliary import write_types
import f2py_interface as respy_f2py

from respy_smm.auxiliary_depreciation import x_all_econ_new_to_old
from respy_smm.tests.auxiliary import get_observed_sample, get_random_init
from respy_smm.tests.auxiliary import get_random_point


def test_1():
    """This test ensures that simple requests are properly handled."""

    get_random_init()
    get_observed_sample()

    est_obj = MaximumLikelihoodEstimationCls(*('test.respy.ini', 3))
    est_obj.evaluate(get_random_point())

    est_obj.terminate(is_gentle=True)


def test_2():
    """This test ensures that ... """

    get_random_init({'flag_interpolation': False})

    get_observed_sample()

    respy_obj = respy_obj_from_new_init('test.respy.ini')

    labels = list()
    labels += ['num_periods', 'seed_emax', 'num_draws_emax', 'num_agents_sim', 'num_types',
               'edu_spec', 'optim_paras', 'is_myopic', "num_paras", 'tau', 'num_draws_prob',
               'num_agents_est', 'seed_prob']

    num_periods, seed_emax, num_draws_emax, num_agents_sim, num_types, edu_spec, optim_paras, \
    is_myopic, num_paras, tau, num_draws_prob, num_agents_est, seed_prob = \
        dist_class_attributes(respy_obj, *labels)

    type_spec_shares = optim_paras["type_shares"]
    type_spec_shifts = optim_paras["type_shifts"]

    max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
    write_types(type_spec_shares, num_agents_sim)
    write_edu_start(edu_spec, num_agents_sim)
    write_draws(num_periods, max_draws)
    write_lagged_start(num_agents_sim)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax, True)
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob, True)

    est_obj = MaximumLikelihoodEstimationCls(*('test.respy.ini', 3))
    state_space_args = list(est_obj.criterion_function.args[11:15])
    data_array = est_obj.criterion_function.args[6]

    num_obs_agent = [num_periods] * num_agents_est

    args = [False, num_draws_emax, num_periods, HUGE_FLOAT, is_myopic, True, data_array,
            num_draws_prob, tau, periods_draws_emax, periods_draws_prob] + state_space_args + \
           [num_agents_est, num_obs_agent, num_types, edu_spec['start'], edu_spec['max'],
            edu_spec['share'], type_spec_shares, type_spec_shifts, num_paras]

    x_all_econ = est_obj.x_all_econ['start']
    fval = respy_f2py.wrapper_criterion(x_all_econ_new_to_old(x_all_econ), *args)
    np.testing.assert_equal(fval == est_obj.info['fval'][0], True)


def test_3():
    """This test ensures that a simple restart of the estimation is working from a checkpoint."""
    # TODO: In principle this also needs to hold for the SMM routine. However, this is only
    #  feasible after the NORPY refactoring as right now part of the randomness is not properly
    #  controlled for.
    get_random_init()
    get_observed_sample()

    random_point = get_random_point()

    est_obj_start = MaximumLikelihoodEstimationCls(*('test.respy.ini', 3))
    est_obj_start.evaluate(random_point)
    est_obj_start.terminate(is_gentle=True)

    est_obj_step = MaximumLikelihoodEstimationCls(*('step.estimagic.ini', 3))
    est_obj_step.evaluate(random_point)
    est_obj_step.terminate(is_gentle=True)

    np.testing.assert_equal(est_obj_start.info['fval'], est_obj_step.info['fval'])
