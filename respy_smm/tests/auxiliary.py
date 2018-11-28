import copy

from scipy.stats import wishart
import numpy as np

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.tests.codes.auxiliary import write_lagged_start
from respy.tests.codes.auxiliary import write_edu_start
from respy.tests.codes.auxiliary import write_draws
from respy.tests.codes.auxiliary import write_types
from respy_smm.moments import get_moments
from respy import RespyCls
from respy import simulate


def get_ingredients(fname):

    respy_obj = RespyCls(fname)

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
    moments_obs = get_moments(simulate(respy_sim)[1])
    num_moments = 0
    for group in ['Choice Probability', 'Wage Distribution']:
        for period in range(num_periods):
            if period not in moments_obs[group].keys():
                continue
            num_moments += len(moments_obs[group][period])

    weighing_matrix = wishart.rvs(num_moments, 0.01 * np.identity(num_moments))

    return respy_obj, moments_obs, num_moments, weighing_matrix
