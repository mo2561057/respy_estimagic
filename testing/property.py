import pickle as pkl
import sys
import os


import numpy as np

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_BUILD

sys.path.insert(0, os.environ["OV_PROJECT"] + "/respy/respy/tests")
sys.path.insert(0, TEST_RESOURCES_BUILD)
import respy
from codes.auxiliary import write_lagged_start
from codes.random_init import generate_random_dict, generate_init
from respy.python.shared.shared_auxiliary import print_init_dict
from codes.auxiliary import write_edu_start
from codes.auxiliary import write_draws
from codes.auxiliary import write_types
import random
from respy import RespyCls
import copy
from smmrespy import run_scipy

from smmrespy.moments import get_moments
from scipy.stats import wishart


while True:
    seed = random.randrange(1, 100000)

    np.random.seed(seed)
    print('seed', seed)

    constr = dict()
    constr['flag_interpolation'] = False
    constr['flag_myopic'] = False # otherwise delta exacltyl on bounds

    dict_ = generate_random_dict(constr)
     # TODO: Thi might not be necessary after all
    dict_['ESTIMATION']['agents'] = dict_['SIMULATION']['agents']

    print_init_dict(dict_)
    # TODO:  I need to ensure that the number of agents in teh sim and est step are identifal

    respy_obj = RespyCls('test.respy.ini')

    num_periods, edu_spec, optim_paras, num_draws_emax, num_agents_sim, num_draws_prob,  \
    num_types = dist_class_attributes(respy_obj, 'num_periods', 'edu_spec', 'optim_paras',
        'num_draws_emax', 'num_agents_sim', 'num_draws_prob', 'num_types')

    max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
    write_types(optim_paras['type_shares'], num_agents_sim)
    write_edu_start(edu_spec, num_agents_sim)
    write_draws(num_periods, max_draws)
    write_lagged_start(num_agents_sim)

    # We need to simulate with a single core.
    respy_sim = copy.deepcopy(respy_obj)
    respy_sim.attr['num_procs'] = 1
    moments_obs = get_moments(respy.simulate(respy_sim)[1])
    num_moments = 0
    for group in ['Choice Probability', 'Wage Distribution']:
        for period in range(num_periods):
            if period not in moments_obs[group].keys():
                continue
            num_moments += len(moments_obs[group][period])

    weighing_matrix = wishart.rvs(num_moments, 0.01 * np.identity(num_moments))
    run_scipy('test.respy.ini', moments_obs, weighing_matrix, 'POWELL', 10)

    rslt = pkl.load(open('smm_monitoring.pkl', 'rb'))
    np.testing.assert_almost_equal(rslt['Step'].ix[0], 0.0)

    # TODO: It woule be better if here this was just a random point of evaluation and not jsut
    # startw ith a different smaple size toe nsure nonzero.

    constr = dict()
    constr['flag_interpolation'] = False
    constr['flag_myopic'] = False

    dict_ = generate_init(constr)

    respy_obj = RespyCls('test.respy.ini')

    num_periods, edu_spec, optim_paras, num_draws_emax, num_agents_sim, num_draws_prob, \
    num_types = dist_class_attributes(respy_obj, 'num_periods', 'edu_spec', 'optim_paras',
        'num_draws_emax', 'num_agents_sim', 'num_draws_prob', 'num_types')

    max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
    write_types(optim_paras['type_shares'], num_agents_sim)
    write_edu_start(edu_spec, num_agents_sim)
    write_draws(num_periods, max_draws)
    write_lagged_start(num_agents_sim)

    # We need to simulate with a single core.
    respy_sim = copy.deepcopy(respy_obj)
    respy_sim.attr['num_procs'] = 1
    moments_obs = get_moments(respy.simulate(respy_sim)[1])
    num_moments = 0
    for group in ['Choice Probability', 'Wage Distribution']:
        for period in range(num_periods):
            if period not in moments_obs[group].keys():
                continue
            num_moments += len(moments_obs[group][period])

    weighing_matrix = wishart.rvs(num_moments, 0.01 * np.identity(num_moments))

    dict_['SIMULATION']['agents'] = np.random.random_integers(1, num_agents_sim)

    for version in ['PYTHON', 'FORTRAN']:
        dict_['PROGRAM']['version'] = version
        if version == 'PYTHON':
            dict_['PROGRAM']['procs'] = 1
        else:
            dict_['PROGRAM']['procs'] = np.random.randint(1, 5)

        print_init_dict(dict_)

        run_scipy('test.respy.ini', moments_obs, weighing_matrix, 'POWELL', 1)

        rslt = pkl.load(open('smm_monitoring.pkl', 'rb'))
        print(rslt['Step'].ix[0])
        #np.testing.assert_almost_equal(rslt['Step'].ix[0], 0.0)



