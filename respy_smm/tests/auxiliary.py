from functools import partial
import sys

from scipy.stats import wishart
import pandas as pd
import numpy as np

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_BUILD
from respy.python.shared.shared_constants import DATA_FORMATS_SIM
from respy.python.shared.shared_constants import DATA_LABELS_SIM
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import MISSING_INT
from respy.tests.codes.auxiliary import write_lagged_start
from respy.tests.codes.auxiliary import write_edu_start
from respy_smm.auxiliary import moments_dict_to_list
from respy.tests.codes.auxiliary import write_draws
from respy.tests.codes.auxiliary import write_types
from respy_smm.auxiliary import smm_sample_pyth
from respy_smm.auxiliary import smm_sample_f2py
from respy_smm.moments import get_moments
from respy import RespyCls

sys.path.insert(0, TEST_RESOURCES_BUILD)
import f2py_interface as respy_f2py


def get_ingredients(fname):

    respy_obj = RespyCls(fname)

    labels = list()
    labels += ['num_periods', 'edu_spec', 'optim_paras', 'num_draws_emax', 'num_agents_sim']
    labels += ['num_draws_prob', 'num_types', 'is_debug', 'seed_emax', 'seed_sim', 'version']

    num_periods, edu_spec, optim_paras, num_draws_emax, num_agents_sim, num_draws_prob, \
        num_types, is_debug, seed_emax, seed_sim, version = \
            dist_class_attributes(respy_obj, *labels)


    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax, is_debug)
    periods_draws_sims = create_draws(num_periods, num_agents_sim, seed_sim, is_debug)
    disturbances = (periods_draws_emax, periods_draws_sims)

    # We want to maintain a pure PYTHON version for testing purposes.
    if version in ['FORTRAN']:
        args = []
        args += [num_periods, num_types, edu_spec['start'], edu_spec['max']]
        args += [edu_spec['max'] + 1]
        state_space_info = respy_f2py.wrapper_create_state_space(*args)
        func = partial(smm_sample_f2py, state_space_info, disturbances, MISSING_INT)
    elif version in ['PYTHON']:
        args = [num_periods, num_types, edu_spec]
        state_space_info = pyth_create_state_space(*args)
        func = partial(smm_sample_pyth, state_space_info, disturbances)
    else:
        raise NotImplementedError

    mat_smm = func(respy_obj)

    data_frame = pd.DataFrame(replace_missing_values(mat_smm), columns=DATA_LABELS_SIM)
    data_frame = data_frame.astype(DATA_FORMATS_SIM)
    data_frame.set_index(['Identifier', 'Period'], drop=False, inplace=True)

    moments_obs = get_moments(data_frame)
    num_moments = len(moments_dict_to_list(moments_obs))

    weighing_matrix = wishart.rvs(num_moments, 0.01 * np.identity(num_moments))

    return respy_obj, moments_obs, num_moments, weighing_matrix
