"""This module contains some tests for the SMM procedures."""
import numpy as np
import pytest

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.tests.codes.random_init import generate_init
from respy.python.process.process_python import process
import respy

from respy_smm.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.auxiliary import transform_unconstraint_to_constraint
from respy_smm.auxiliary import transform_constraint_to_unconstraint
from respy_smm.auxiliary import get_econ_from_optim
from respy_smm.auxiliary import get_optim_from_econ
from respy_smm.weighing import get_weighing_matrix
from respy_smm.auxiliary import apply_scaling
from respy_smm.auxiliary import get_scales
from respy_smm.moments import get_moments


def test_1():
    """This function tests the back-and-forth transformations required due to the bounds
    specified for the parameters."""
    for _ in range(10000):
        value = np.random.uniform()

        lower = np.random.choice([value - np.random.uniform(0.01, 1), None])
        upper = np.random.choice([value + np.random.uniform(0.01, 1), None])

        constraint = transform_unconstraint_to_constraint(value, [lower, upper])
        stat = transform_constraint_to_unconstraint(constraint, [lower, upper])

        np.testing.assert_almost_equal(value, stat)


def test_2():
    """This function tests the back-and-forth transformation of the economic and optimization
    parameters."""
    for _ in range(100):

        generate_init()

        respy_base = respy.RespyCls('test.respy.ini')

        optim_paras, num_paras = dist_class_attributes(respy_base, 'optim_paras', 'num_paras')

        x_all_optim = get_optim_paras(optim_paras, num_paras, 'all', True)
        x_all_econ = get_econ_from_optim(x_all_optim, optim_paras['paras_bounds'])
        stat = get_optim_from_econ(x_all_econ, optim_paras['paras_bounds'])

        np.testing.assert_almost_equal(x_all_optim, stat)


@pytest.mark.skipif(True, reason='unknown at this point')
def test_3():
    """This function tests the robustness of the evaluation of the criterion function for
    random requests."""
    for _ in range(5):

        constr = dict()
        constr['flag_estimation'] = True
        generate_init(constr)

        respy_base = respy.RespyCls('test.respy.ini')
        respy.simulate(respy_base)

        df_base = process(respy_base)
        num_agents_base = df_base.index.get_level_values('Identifier').nunique()
        num_agents_smm = np.random.randint(1, num_agents_base)

        weighing_matrix = get_weighing_matrix(df_base, 10, num_agents_smm)

        moments_obs = get_moments(process(respy_base))

        # We now set up the SMM estimation.
        est_obj = SimulationBasedEstimationCls('test.respy.ini', moments_obs, weighing_matrix)

        # Sample some evaluation points ane evaluation the criterion function, where we want to
        # trace out extremes of the evaluation function.
        num_free = respy_base.attr['optim_paras']['paras_fixed'].count(False)
        if np.random.choice([True, False]):
            x_free_optim = np.random.uniform(size=num_free)
        else:
            x_free_optim = np.random.normal(size=num_free)

        est_obj.criterion(False, x_free_optim)


def test_4():
    """This function simply tests the scaling setup."""
    for _ in range(100):

        generate_init()
        respy_base = respy.RespyCls('test.respy.ini')

        optim_paras, num_paras = dist_class_attributes(respy_base, 'optim_paras', 'num_paras')
        x_free_optim_start = get_optim_paras(optim_paras, num_paras, 'free', True)
        scales = get_scales(x_free_optim_start)
        rslt = apply_scaling(x_free_optim_start,  scales, 'do')

        np.testing.assert_almost_equal(apply_scaling(rslt,  scales, 'undo'), x_free_optim_start)


def test_5():
    """This function simply tests the construction of the weighing matrix."""
    num_boots = np.random.randint(1, 10)

    generate_init()
    respy_base = respy.RespyCls('test.respy.ini')
    respy.simulate(respy_base)

    df_base = process(respy_base)
    num_agents_base = df_base.index.get_level_values('Identifier').nunique()
    num_agents_smm = np.random.randint(1, num_agents_base)

    get_weighing_matrix(df_base, num_boots, num_agents_smm)
