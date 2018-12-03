#!/usr/bin/env python
import pickle as pkl
import numpy as np

import sys

sys.path.insert(0, '../')

import respy_smm


np.random.seed(123)

# This is some setup required for the Monte Carlo exercise.
if False:

    from respy_smm import get_weighing_matrix
    from respy.python.process.process_python import process
    from respy_smm import get_moments
    import respy

    respy_obs = respy.RespyCls('debug.respy.ini')

    respy.simulate(respy_obs)

    df_base = process(respy_obs)

    num_agents_smm = 10
    num_boots = 50

    weighing_matrix = get_weighing_matrix(df_base, num_boots, num_agents_smm, True)
    moments_obs = get_moments(df_base, True)


weighing_matrix = pkl.load(open('weighing.respy.pkl', 'rb'))
moments_obs = pkl.load(open('moments.respy.pkl', 'rb'))

toolbox = 'nag'

toolbox_spec = dict()
toolbox_spec['max_evals'] = 1
toolbox_spec['algorithm'] = 'bobyqa'


respy_smm.optimize('debug.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)
