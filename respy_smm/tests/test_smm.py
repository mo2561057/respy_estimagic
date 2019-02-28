"""This module contains some tests for the SMM procedures."""
import numpy as np

from respy.pre_processing.data_processing import process_dataset
from respy.tests.codes.random_init import generate_init
import respy

from respy_smm.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.weighing import get_weighing_matrix
from respy_smm.moments import get_moments


def test_1():
    """This function tests the robustness of the evaluation of the criterion function for
    random requests."""
    for _ in range(5):

        constr = dict()

        # Otherwise we need to execute the script with MPIEXEC.
        constr['flag_parallelism_mpi'] = False
        generate_init(constr)

        respy_base = respy.RespyCls('test.respy.ini')
        respy_base.simulate()

        df_base = process_dataset(respy_base)
        num_agents_base = df_base.index.get_level_values('Identifier').nunique()
        num_agents_smm = np.random.randint(1, num_agents_base)

        weighing_matrix = get_weighing_matrix(df_base, 10, num_agents_smm)

        moments_obs = get_moments(process_dataset(respy_base))

        # We now set up the SMM estimation.
        est_obj = SimulationBasedEstimationCls('test.respy.ini', moments_obs, weighing_matrix)

        # TODO: It would be nice to allow for more general points of evaluation than just the
        #  starting value.
        paras_free = ~np.array(respy_base.get_attr('optim_paras')['paras_fixed'])
        x_all_econ_start = est_obj.get_attr('x_all_econ_start')
        x_free_econ_start = x_all_econ_start[paras_free]

        est_obj.criterion(x_free_econ_start)
