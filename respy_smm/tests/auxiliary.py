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
from respy_smm.MaximumLikelihoodEstimation import MaximumLikelihoodEstimationCls
from respy_smm.SimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.auxiliary_depreciation import respy_obj_from_new_init
from respy.python.simulate.simulate_auxiliary import write_out

from respy_smm.auxiliary import smm_sample_f2py
from respy import RespyCls
from respy_smm.auxiliary_depreciation import respy_ini_old_to_new
from respy.tests.codes.random_init import generate_init
import os
import copy
from respy.tests.codes.random_init import write_init_file

sys.path.insert(0, TEST_RESOURCES_BUILD)
import f2py_interface as respy_f2py



def smm_sample_pyth(state_space_info, disturbances, respy_obj):
    """This function is a wrapper that is supposed to facilitate the application of SMM
    estimation for the RESPY package."""
    states_all, states_number_period, mapping_state_idx, max_states_period = state_space_info
    periods_draws_emax, periods_draws_sims = disturbances

    labels = list()
    labels += ['num_periods', 'edu_spec', 'optim_paras', 'num_draws_emax', 'is_debug']
    labels += ['is_interpolated', 'num_points_interp', 'is_myopic', 'num_agents_sim', 'seed_sim']
    labels += ['file_sim', 'num_types', 'is_myopic']

    num_periods, edu_spec, optim_paras, num_draws_emax, is_debug, is_interpolated, \
        num_points_interp, is_myopic, num_agents_sim, seed_sim, file_sim, num_types, is_myopic = \
        dist_class_attributes(respy_obj, *labels)

    args = (num_periods, states_number_period, states_all, max_states_period, optim_paras)
    periods_rewards_systematic = pyth_calculate_rewards_systematic(*args)

    args = (num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax,
        states_number_period, periods_rewards_systematic, mapping_state_idx, states_all,
        is_debug, is_interpolated, num_points_interp, edu_spec, optim_paras, file_sim, False)
    periods_emax = pyth_backward_induction(*args)

    args = (periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, num_periods,
        num_agents_sim, periods_draws_sims, seed_sim, file_sim, edu_spec, optim_paras,
        num_types, is_debug)
    dat = pyth_simulate(*args)

    return dat


def get_random_point(fname='test.respy.ini'):
    respy_base = respy_obj_from_new_init(fname)
    respy_base.paras_free = ~np.array(respy_base.get_attr('optim_paras')['paras_fixed'])
    values = list()
    bounds = respy_base.get_attr('optim_paras')['paras_bounds']
    for i, is_free in enumerate(respy_base.paras_free):
        if not is_free:
            continue
        lower, upper = [-99 * (-1) ** i if e is None else e for i, e in enumerate(bounds[i])]
        values.append(float(np.random.uniform(lower, upper, 1)))

    return np.array(values)


def get_random_init(constr=dict()):
    """This is a wrapper around the RESPY functionality."""
    if 'PMI_SIZE' not in os.environ.keys():
        constr['flag_parallelism_mpi'] = False

    # There are some keys that are not part of the RESPY constraints, so these need to be removed
    # first.
    constr_respy = copy.deepcopy(constr)
    constr_respy.pop('flag_agents_equality', None)
    init_dict = generate_init(constr_respy)

    # This constraint is not part of the original RESPY coded.
    if 'flag_agents_equality' in constr.keys():
        if constr['flag_agents_equality'] is True:
            init_dict['ESTIMATION']['agents'] = init_dict['SIMULATION']['agents']

    write_init_file(init_dict)

    file_name = 'test.respy.ini'
    respy_ini_old_to_new(file_name, True, file_name)


def get_observed_sample(fname='test.respy.ini'):
    """This method simulates a sample based on the initialization file to serve as the
    observed dataset during testing.

    We are not using the RESPY capabilities directly as this results in conflicts in case of
    parallel executions with the nested MPIEXEC calls.
    """
    respy_base = respy_obj_from_new_init(fname)

    labels = list()
    labels += ['num_procs', 'num_periods', 'is_debug', 'seed_emax', 'seed_sim']
    labels += ['num_draws_emax', 'num_agents_sim', 'num_types', 'edu_spec', 'version']

    num_procs, num_periods, is_debug, seed_emax, seed_sim, num_draws_emax, num_agents_sim, \
    num_types, edu_spec, version = dist_class_attributes(respy_base, *labels)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax, is_debug)
    periods_draws_sims = create_draws(num_periods, num_agents_sim, seed_sim, is_debug)

    disturbances = (periods_draws_emax, periods_draws_sims)

    # We want to maintain a pure PYTHON version for testing purposes.
    args = list()
    args += [num_periods, num_types, edu_spec['start'], edu_spec['max'], edu_spec['max'] + 1]
    state_space_info = respy_f2py.wrapper_create_state_space(*args)

    simulate_sample = partial(smm_sample_f2py, state_space_info, disturbances, -99)

    data_array = replace_missing_values(simulate_sample(respy_base))
    data_frame = pd.DataFrame(data_array, columns=DATA_LABELS_SIM)
    data_frame = data_frame.astype(DATA_FORMATS_SIM)
    data_frame.set_index(["Identifier", "Period"], drop=False, inplace=True)

    write_out(respy_base, data_frame)

    return data_frame


def mock_get_weighing_matrix(df):
    num_moments = (df['Period'].max() + 1) * 4
    df = num_moments
    scale = np.identity(num_moments)
    return wishart.rvs(df, scale, size=1)


def mock_get_moments(df):
    moments = dict()
    moments['Choice Probability'] = dict()

    info = df['Choice'].groupby('Period').value_counts(normalize=True).to_dict()
    for period in sorted(df['Period'].unique().tolist()):
        moments['Choice Probability'][period] = []
        for choice in range(1, 5):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments['Choice Probability'][period].append(stat)
    return moments


def run_regression_test(seed):

    np.random.seed(seed)

    # Generate a new regression vault ...
    get_random_init()
    df = get_observed_sample()

    weighing_matrix = mock_get_weighing_matrix(df)
    moments_obs = mock_get_moments(df)

    args = ('test.respy.ini', moments_obs, weighing_matrix, mock_get_moments, 5)
    est_obj = SimulationBasedEstimationCls(*args)
    fval_smm = est_obj.info['fval'][0]
    est_obj.terminate(True)

    est_obj = MaximumLikelihoodEstimationCls(*('test.respy.ini', 3))
    fval_mle = est_obj.info['fval'][0]
    est_obj.terminate(True)
    rslt = (fval_smm, fval_mle)

    return rslt