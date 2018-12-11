#!/usr/bin/env python
import pickle as pkl
import numpy as np
from respy_smm.auxiliary_depreciation import shocks_spec_new_to_old
from respy.pre_processing.model_processing import write_init_file
from respy.pre_processing.model_processing import read_init_file

import sys

sys.path.insert(0, '../')

import respy_smm


np.random.seed(123)

# This is some setup required for the Monte Carlo exercise.
if False:

    from respy_smm import get_weighing_matrix
    from respy.pre_processing.data_processing import process_dataset
    from respy_smm import get_moments
    import respy

    init_dict = read_init_file('debug.respy.ini')
    shock_spec_new = init_dict['SHOCKS']['coeffs']
    shock_spec_old = shocks_spec_new_to_old(shock_spec_new)
    init_dict['SHOCKS']['coeffs'] = shock_spec_old
    write_init_file(init_dict, file_name=".smm.respy.ini")
    respy_obs = respy.RespyCls('.smm.respy.ini')




    respy_obs.simulate()

    df_base = process_dataset(respy_obs)

    num_agents_smm = 10
    num_boots = 50

    weighing_matrix = get_weighing_matrix(df_base, num_boots, num_agents_smm, True)
    moments_obs = get_moments(df_base, True)


weighing_matrix = pkl.load(open('weighing.respy.pkl', 'rb'))
moments_obs = pkl.load(open('moments.respy.pkl', 'rb'))

toolbox = 'nag'

toolbox_spec = dict()
toolbox_spec['max_evals'] = 2
toolbox_spec['algorithm'] = 'bobyqa'


respy_smm.optimize('debug.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)
