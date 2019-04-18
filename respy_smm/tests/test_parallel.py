""" This module contains straightforward tests that ensure that the number of processors does not
affect the values of the criterion function.
"""
import os

import numpy as np
import pytest

from respy_smm.MaximumLikelihoodEstimation import MaximumLikelihoodEstimationCls
from respy_smm.SimulationBasedEstimation import SimulationBasedEstimationCls
from respy.pre_processing.model_processing import write_init_file
from respy_smm.auxiliary_depreciation import respy_ini_old_to_new
from respy_smm.tests.auxiliary import mock_get_weighing_matrix
from respy_smm.tests.auxiliary import get_observed_sample
from respy_smm.tests.auxiliary import get_random_point
from respy_smm.tests.auxiliary import get_random_init
from respy_smm.tests.auxiliary import mock_get_moments


@pytest.mark.skipif('PMI_SIZE' not in os.environ.keys(), reason="requires mpiexec")
def test_1():

    init_dict_old = get_random_init()
    df = get_observed_sample()
    weighing_matrix = mock_get_weighing_matrix(df)
    moments_obs = mock_get_moments(df)
    point = get_random_point()

    rslt = dict()
    rslt['mle'], rslt['smm'] = list(), list()

    for num_procs in [1, 3]:
        init_dict_old['PROGRAM']['procs'] = num_procs

        write_init_file(init_dict_old)
        respy_ini_old_to_new('test.respy.ini', True, 'test.respy.ini')

        est_obj = MaximumLikelihoodEstimationCls(*('test.respy.ini', 3))
        est_obj.evaluate(point)
        est_obj.terminate(is_gentle=True)

        rslt['mle'].append(est_obj.fval['current'])

        args = ('test.respy.ini', moments_obs, weighing_matrix, mock_get_moments, 5)
        est_obj = SimulationBasedEstimationCls(*args)
        est_obj.evaluate(point)
        est_obj.terminate(is_gentle=True)

        rslt['smm'].append(est_obj.fval['current'])

    for key_ in rslt.keys():
        np.testing.assert_almost_equal(*rslt[key_])
