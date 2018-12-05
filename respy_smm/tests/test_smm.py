"""This module contains some tests for the SMM procedures."""
import numpy as np
import pytest

from respy.pre_processing.data_processing import process_dataset
from respy.tests.codes.random_init import generate_init
import respy

from respy_smm.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.weighing import get_weighing_matrix
from respy_smm.moments import get_moments


@pytest.mark.skipif(True, reason='need random evaluation point')
def test_1():
    """This function tests the robustness of the evaluation of the criterion function for
    random requests."""
    for _ in range(5):

        constr = dict()
        constr['flag_estimation'] = True
        generate_init(constr)

        respy_base = respy.RespyCls('test.respy.ini')
        respy.simulate(respy_base)

        df_base = process_dataset(respy_base)
        num_agents_base = df_base.index.get_level_values('Identifier').nunique()
        num_agents_smm = np.random.randint(1, num_agents_base)

        weighing_matrix = get_weighing_matrix(df_base, 10, num_agents_smm)

        moments_obs = get_moments(process_dataset(respy_base))

        # We now set up the SMM estimation.
        est_obj = SimulationBasedEstimationCls('test.respy.ini', moments_obs, weighing_matrix)

        # Sample some evaluation points ane evaluation the criterion function, where we want to
        # trace out extremes of the evaluation function.
        num_free = respy_base.attr['optim_paras']['paras_fixed'].count(False)
        if np.random.choice([True, False]):
            x_free_optim = np.random.uniform(size=num_free)
        else:
            x_free_optim = np.random.normal(size=num_free)

        # TODO: We are only allowing for economic parameters, so we need a routine that simulates
        # a valid point of evaluation.
        est_obj.criterion(x_free_optim)


def test_2():
    """This function simply tests the construction of the weighing matrix."""
    num_boots = np.random.randint(1, 10)

    generate_init()
    respy_base = respy.RespyCls('test.respy.ini')
    respy_base.simulate()

    df_base = process_dataset(respy_base)
    num_agents_base = df_base.index.get_level_values('Identifier').nunique()
    num_agents_smm = np.random.randint(1, num_agents_base)

    get_weighing_matrix(df_base, num_boots, num_agents_smm)
