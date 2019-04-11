"""This module contains some tests for the SMM procedures."""
import numpy as np

from respy_smm.SimulationBasedEstimation import SimulationBasedEstimationCls
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
    # TODO> WHy not needed to constrain to same sizse sim sample and eset sampe
    get_random_init(constr)

    df = get_observed_sample()
    weighing_matrix = mock_get_weighing_matrix(df)
    moments_obs = mock_get_moments(df)

    args = ('test.respy.ini', moments_obs, weighing_matrix, mock_get_moments, 5)
    est_obj = SimulationBasedEstimationCls(*args)
    est_obj.terminate(is_gentle=True)

    np.testing.assert_equal(est_obj.fval['start'], 0.00)