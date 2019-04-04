#!/usr/bin/env python
"""This module currently holds the first attempt to reactivate the MLE estimation for the Option
value estimation.

    TODO:
        * We need to set up basic property testing and a Monte Carlo setup.
        * We need to be able to run this in parallel (including tests)
        * This appears to achieve the separation between the criterion function and the
        optimization routine.

"""
import pickle as pkl
import os

import numpy as np

from respy_smm.MaximumLikelihoodEstimation import MaximumLikelihoodEstimationCls
from respy_smm.SimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.auxiliary_depreciation import respy_obj_from_new_init
from respy.tests.codes.auxiliary import simulate_observed
from respy.pre_processing.data_processing import process_dataset
from respy_smm.optimizers.auxiliary_pybobyqa import wrapper_pybobyqa, get_box_bounds
from smm_preparations import get_moments


def prepare_debugging_setup(init_file):
    from smm_preparations import get_weighing_matrix
    from smm_preparations import get_moments

    respy_obj = respy_obj_from_new_init(init_file)
    if not os.path.exists('data.respy.dat'):
        respy_obj = simulate_observed(respy_obj)
        df_base = process_dataset(respy_obj)

    if not os.path.exists('moments.respy.pkl'):
        moments_obs = get_moments(df_base, True)
    else:
        moments_obs = pkl.load(open('moments.respy.pkl', 'rb'))

    if not os.path.exists('weighing.respy.pkl'):
        num_agents_smm, num_boots = 10, 50
        weighing_matrix = get_weighing_matrix(df_base, num_boots, num_agents_smm, True)
    else:
        weighing_matrix = pkl.load(open('weighing.respy.pkl', 'rb'))

    return moments_obs, weighing_matrix


np.random.seed(213)

# We need to make sure we have a clean version of the library compiled.
os.chdir('../')
#os.system('git clean -df')
os.chdir('working')

# We need to specify the basics of the optimization problem.
init_file, max_evals = 'debug.respy.ini', 2
moments_obs, weighing_matrix = prepare_debugging_setup(init_file)
# --------------------------------------------------------------------------------------------------
# GENERAL setup for optimization problems, https://en.wikipedia.org/wiki/Adapter_pattern
# --------------------------------------------------------------------------------------------------
print('\n I need to add ability to restart \n')

args = (init_file, moments_obs, weighing_matrix, get_moments, max_evals)
adapter_smm = SimulationBasedEstimationCls(*args)

args = (init_file, max_evals)
adapter_mle = MaximumLikelihoodEstimationCls(*args)

for est_obj in [adapter_smm, adapter_mle]:
    # ----------------------------------------------------------------------------------------------
    # SPECIFIC setup for the optimizer
    # ----------------------------------------------------------------------------------------------
    box = get_box_bounds(init_file)

    kwargs = dict()
    kwargs['scaling_within_bounds'] = True
    kwargs['bounds'] = (box[:, 0], box[:, 1])
    kwargs['objfun_has_noise'] = True
    kwargs['maxfun'] = 10e6

    rslt = wrapper_pybobyqa(est_obj.evaluate, est_obj.x_free_econ_start, **kwargs)

    print('finished estimation \n')
