"""This module contains supporting functions for the SMM estimation."""
import os

import numpy as np

from respy.pre_processing.data_processing import process_dataset
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.fortran.interface import write_resfort_initialization
from respy.python.shared.shared_constants import ROOT_DIR, HUGE_FLOAT
from respy_smm.auxiliary_depreciation import respy_obj_from_new_init
from respy.python.simulate.simulate_auxiliary import get_random_types
from respy.python.simulate.simulate_auxiliary import get_random_edu_start, \
    get_random_lagged_start

from respy_smm.src import smm_interface


def get_mpi():
    """This function returns the MPI connector (if possible)"""
    try:
        from mpi4py import MPI
        return MPI
    except ImportError:
        return None


def format_column(x):
    """This function provides pretty floats for the columns."""
    if isinstance(x, str):
        return '{}'.format(x)
    else:
        return '{:25.5f}'.format(x)


def smm_sample_f2py(state_space_info, initial_conditions, disturbances, slavecomm_f2py, respy_obj):
    """This function is a wrapper that is supposed to facilitate the application of SMM
    estimation for the RESPY package."""

    sample_edu_start, sample_lagged_start = initial_conditions
    periods_draws_emax, periods_draws_sims = disturbances

    labels = list()
    labels += ['num_periods', 'edu_spec', 'optim_paras', 'num_draws_emax', 'is_debug']
    labels += ['is_interpolated', 'num_points_interp', 'is_myopic', 'num_agents_sim', "num_paras"]
    labels += ['num_procs', 'num_types', 'seed_sim']

    num_periods, edu_spec, optim_paras, num_draws_emax, is_debug, is_interpolated, \
        num_points_interp, is_myopic, num_agents_sim, num_paras, num_procs, num_types, seed_sim = \
        dist_class_attributes(respy_obj, *labels)

    np.random.seed(seed_sim)

    shocks_cholesky = optim_paras['shocks_cholesky']
    coeffs_common = optim_paras['coeffs_common']
    coeffs_home = optim_paras['coeffs_home']
    coeffs_edu = optim_paras['coeffs_edu']
    coeffs_a = optim_paras['coeffs_a']
    coeffs_b = optim_paras['coeffs_b']
    delta = optim_paras['delta']

    type_spec_shares = optim_paras['type_shares']
    type_spec_shifts = optim_paras['type_shifts']

    args = (num_types, optim_paras, num_agents_sim, sample_edu_start, is_debug)
    sample_types = np.ones(num_agents_sim)

    args = state_space_info + (coeffs_common, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, delta, is_interpolated, num_points_interp, num_draws_emax, num_periods,
        is_myopic, is_debug, periods_draws_emax, num_agents_sim, periods_draws_sims,
        type_spec_shares, type_spec_shifts, edu_spec['start'], edu_spec['max'], edu_spec['lagged'],
        edu_spec['share'], num_paras, sample_edu_start, sample_lagged_start, sample_types,
        slavecomm_f2py)

    dat = smm_interface.wrapper_smm(*args)

    return dat


def get_communicator(respy_obj, data_array=None):
    """This is a temporary function that sets up the communicator."""

    # There is no data available for the SMM estimation, so we generate a random sample that
    # eases the all code that is coming later.
    if data_array is None:
        data_array = np.random.uniform(size=64).reshape(8, 8)

    labels = list()
    labels += ['optim_paras', 'num_periods', 'edu_spec', 'is_debug', 'num_draws_emax']
    labels += ['seed_emax', 'is_interpolated', 'num_points_interp', 'is_myopic', 'tau']
    labels += ['num_procs', 'num_agents_sim', 'num_draws_prob', 'seed_prob', 'seed_sim']
    labels += ['optimizer_options',  'optimizer_used', 'maxfun', 'precond_spec', 'file_sim']
    labels += ['num_paras', 'num_types', 'num_agents_est']

    optim_paras, num_periods, edu_spec, is_debug, num_draws_emax, seed_emax, is_interpolated, \
        num_points_interp, is_myopic, tau, num_procs, num_agents_sim, num_draws_prob, \
        seed_prob, seed_sim, optimizer_options, optimizer_used, maxfun, precond_spec, \
        file_sim, num_paras, num_types, num_agents_est = dist_class_attributes(respy_obj, *labels)

    args = (optim_paras, is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_spec, is_debug, num_draws_prob, num_agents_sim, seed_prob, seed_emax,
        tau, num_procs, 'simulate', seed_sim, optimizer_options, optimizer_used, maxfun, num_paras,
        precond_spec, file_sim, data_array, num_types, num_agents_est)

    write_resfort_initialization(*args)

    MPI = get_mpi()
    info = MPI.Info.Create()
    info.Set('wdir', os.getcwd())

    test = ROOT_DIR + '/.bld/fortran/resfort_slave'
    worker = MPI.COMM_SELF.Spawn(test, info=info, maxprocs=num_procs - 1)

    return worker


def is_valid_covariance_matrix(shocks_coeffs_new):
    sds, rho = shocks_coeffs_new[:4], shocks_coeffs_new[4:]

    shocks_cov = np.zeros((4, 4))

    shocks_cov[1, 0] = rho[0] * sds[1] * sds[0]
    shocks_cov[2, 0] = rho[1] * sds[2] * sds[0]
    shocks_cov[2, 1] = rho[2] * sds[2] * sds[1]
    shocks_cov[3, 0] = rho[3] * sds[3] * sds[0]
    shocks_cov[3, 1] = rho[4] * sds[3] * sds[1]
    shocks_cov[3, 2] = rho[5] * sds[3] * sds[2]

    np.fill_diagonal(shocks_cov, sds ** 2)

    shocks_cov = shocks_cov + shocks_cov.T - np.diag(shocks_cov.diagonal())
    try:
        np.linalg.cholesky(shocks_cov)
        return True
    except np.linalg.linalg.LinAlgError:
        return False


def get_processed_dataset(init_file):
    respy_obj = respy_obj_from_new_init(init_file)
    data_array = process_dataset(respy_obj).values
    data_array[np.isnan(data_array)] = HUGE_FLOAT
    data_array = np.ascontiguousarray(data_array, np.float64)

    return data_array


def get_initial_conditions(respy_obj):
    # TODO: Cleanup the attribute list, not all needed.
    labels = list()
    labels += ['num_procs', 'num_periods', 'is_debug', 'seed_emax', 'seed_sim']
    labels += ['num_draws_emax', 'num_agents_sim', 'num_types', 'edu_spec', 'version']

    num_procs, num_periods, is_debug, seed_emax, seed_sim, num_draws_emax, num_agents_sim, \
    num_types, edu_spec, version = dist_class_attributes(respy_obj, *labels)

    np.random.seed(seed_sim)
    sample_edu_start = get_random_edu_start(edu_spec, num_agents_sim, is_debug)
    sample_lagged_start = get_random_lagged_start(edu_spec, num_agents_sim, sample_edu_start,
                                                  is_debug)

    return sample_edu_start, sample_lagged_start
