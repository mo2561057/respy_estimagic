"""This module contains some tests for the SMM procedures."""
import numpy as np

from respy_smm.SimulationBasedEstimation import SimulationBasedEstimationCls
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.tests.codes.auxiliary import write_lagged_start
from respy.tests.codes.auxiliary import write_edu_start
from respy.tests.codes.auxiliary import write_draws
from respy.tests.codes.auxiliary import write_types
from respy_smm.auxiliary_depreciation import respy_obj_from_new_init
from respy_smm.tests.auxiliary import get_observed_sample, get_random_point, get_random_init, \
    mock_get_weighing_matrix, mock_get_moments


def test_1():
    """This function tests the robustness of the evaluation of the criterion function for
    random requests."""
    get_random_init()

    df = get_observed_sample()

    weighing_matrix = mock_get_weighing_matrix(df)
    moments_obs = mock_get_moments(df)

    args = ('test.respy.ini', moments_obs, weighing_matrix, mock_get_moments, 5)
    est_obj = SimulationBasedEstimationCls(*args)
    est_obj.evaluate(get_random_point())
    est_obj.terminate(is_gentle=True)


def test_2():

    """The evaluation of the criterion function is zero at the truth."""
    constr = dict()
    constr['flag_agents_equality'] = True
    constr['flag_interpolation'] = False
    constr['version'] = 'FORTRAN'

    get_random_init(constr)

    respy_base = respy_obj_from_new_init('test.respy.ini')

    # Extract class attributes
    num_periods, edu_spec, optim_paras, num_draws_emax, num_agents_sim, num_draws_prob, \
        = dist_class_attributes(respy_base, 'num_periods', "edu_spec", "optim_paras",
            "num_draws_emax", "num_agents_sim", "num_draws_prob")

    type_spec_shares = optim_paras["type_shares"]

    max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
    write_types(type_spec_shares, num_agents_sim)
    write_edu_start(edu_spec, num_agents_sim)
    write_draws(num_periods, max_draws)
    write_lagged_start(num_agents_sim)

    df = get_observed_sample()

    weighing_matrix = mock_get_weighing_matrix(df)
    moments_obs = mock_get_moments(df)

    args = ('test.respy.ini', moments_obs, weighing_matrix, mock_get_moments, 5)
    est_obj = SimulationBasedEstimationCls(*args)
    np.testing.assert_almost_equal(est_obj.fval['start'], 0.00)
    est_obj.terminate(is_gentle=True)
