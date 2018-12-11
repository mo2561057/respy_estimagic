"""This module contains supporting functions for the SMM estimation."""
import os

from respy.python.solve.solve_auxiliary import pyth_calculate_rewards_systematic
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.fortran.interface import write_resfort_initialization
from respy.python.simulate.simulate_python import pyth_simulate
from respy.python.shared.shared_constants import ROOT_DIR

from respy_smm.src import smm_interface

if 'PMI_SIZE' in os.environ.keys():
    try:
        from mpi4py import MPI
    except ImportError:
        pass


def format_column(x):
    """This function provides pretty floats for the columns."""
    if isinstance(x, str):
        return '{}'.format(x)
    else:
        return '{:25.5f}'.format(x)


def moments_dict_to_list(moments_dict):
    """This function constructs a list of available moments based on the moment dictionary."""
    moments_list = []
    for group in ['Choice Probability', 'Wage Distribution']:
        for period in sorted(moments_dict[group].keys()):
            moments_list.extend(moments_dict[group][period])
    return moments_list


def smm_sample_pyth(state_space_info, disturbances, respy_obj):
    """This function is a wrapper that is supposed to facilitate the application of SMM
    estimation for the RESPY package."""
    states_all, states_number_period, mapping_state_idx, max_states_period = state_space_info
    periods_draws_emax, periods_draws_sims = disturbances

    num_periods, edu_spec, optim_paras, num_draws_emax, is_debug, is_interpolated, \
    num_points_interp, is_myopic, num_agents_sim, seed_sim, file_sim, num_types, \
    is_myopic = dist_class_attributes(respy_obj, 'num_periods', 'edu_spec', 'optim_paras',
        'num_draws_emax', 'is_debug', 'is_interpolated', 'num_points_interp', 'is_myopic',
        'num_agents_sim', 'seed_sim', 'file_sim', 'num_types', 'is_myopic')

    args = (num_periods, states_number_period, states_all, max_states_period, optim_paras)
    periods_rewards_systematic = pyth_calculate_rewards_systematic(*args)

    args = (num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax,
        states_number_period, periods_rewards_systematic, mapping_state_idx, states_all,
        is_debug, is_interpolated, num_points_interp, edu_spec, optim_paras, file_sim, False)
    periods_emax = pyth_backward_induction(*args)

    args = (periods_rewards_systematic, mapping_state_idx, periods_emax, states_all,
        num_periods, num_agents_sim, periods_draws_sims, seed_sim, file_sim, edu_spec, optim_paras,
        num_types, is_debug)
    dat = pyth_simulate(*args)

    return dat


def smm_sample_f2py(state_space_info, disturbances, slavecomm_f2py, respy_obj):
    """This function is a wrapper that is supposed to facilitate the application of SMM
    estimation for the RESPY package."""
    periods_draws_emax, periods_draws_sims = disturbances

    num_periods, edu_spec, optim_paras, num_draws_emax, is_debug, is_interpolated, \
    num_points_interp, is_myopic, num_agents_sim, num_paras, num_procs = dist_class_attributes(
        respy_obj, 'num_periods', 'edu_spec', 'optim_paras', 'num_draws_emax', 'is_debug',
        'is_interpolated', 'num_points_interp', 'is_myopic', 'num_agents_sim', "num_paras",
        'num_procs')

    shocks_cholesky = optim_paras['shocks_cholesky']
    coeffs_common = optim_paras['coeffs_common']
    coeffs_home = optim_paras['coeffs_home']
    coeffs_edu = optim_paras['coeffs_edu']
    coeffs_a = optim_paras['coeffs_a']
    coeffs_b = optim_paras['coeffs_b']
    delta = optim_paras['delta']

    type_spec_shares = optim_paras['type_shares']
    type_spec_shifts = optim_paras['type_shifts']

    args = state_space_info + (coeffs_common, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, delta, is_interpolated, num_points_interp, num_draws_emax, num_periods,
        is_myopic, is_debug, periods_draws_emax, num_agents_sim, periods_draws_sims,
        type_spec_shares, type_spec_shifts, edu_spec['start'], edu_spec['max'], edu_spec['lagged'],
        edu_spec['share'], num_paras, slavecomm_f2py)

    dat = smm_interface.wrapper_smm(*args)

    return dat


def get_communicator(respy_obj):
    """This is a temporary function that sets up the communicator."""
    optim_paras, num_periods, edu_spec, is_debug, num_draws_emax, seed_emax, is_interpolated, \
    num_points_interp, is_myopic, tau, num_procs, num_agents_sim, num_draws_prob, \
    seed_prob, seed_sim, optimizer_options, optimizer_used, maxfun, precond_spec, \
    file_sim, num_paras, num_types, num_agents_est = dist_class_attributes(respy_obj,
        'optim_paras', 'num_periods', 'edu_spec', 'is_debug', 'num_draws_emax', 'seed_emax',
        'is_interpolated', 'num_points_interp', 'is_myopic', 'tau', 'num_procs', 'num_agents_sim',
        'num_draws_prob', 'seed_prob', 'seed_sim', 'optimizer_options',  'optimizer_used',
        'maxfun', 'precond_spec', 'file_sim', 'num_paras', 'num_types', 'num_agents_est')

    args = (optim_paras, is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_spec, is_debug, num_draws_prob, num_agents_sim, seed_prob, seed_emax,
        tau, num_procs, 'simulate', seed_sim, optimizer_options, optimizer_used, maxfun, num_paras,
        precond_spec, file_sim, None, num_types, num_agents_est)

    write_resfort_initialization(*args)

    info = MPI.Info.Create()
    info.Set('wdir', os.getcwd())

    test = ROOT_DIR + '/.bld/fortran/resfort_slave'
    worker = MPI.COMM_SELF.Spawn(test, info=info, maxprocs=num_procs - 1)

    return worker
