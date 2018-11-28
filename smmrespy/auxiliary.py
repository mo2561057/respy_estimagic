"""This module contains supporting functions for the SMM estimation."""
from math import floor
from math import log10
from time import sleep
import pickle as pkl
import datetime
import random
import string
import shutil
import uuid
import glob
import os

import numpy as np

if 'PMI_SIZE' in os.environ.keys():
    try:
        from mpi4py import MPI
    except ImportError:
        pass


from respy.python.solve.solve_auxiliary import pyth_calculate_rewards_systematic
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.fortran.interface import write_resfort_initialization
from respy.python.simulate.simulate_python import pyth_simulate
from respy.python.shared.shared_constants import ROOT_DIR
from respy.clsRespy import PARAS_MAPPING
import respy


from smmrespy.src import smm_interface

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '\n ... %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def format_column(x):
    """This function provides pretty floats for the columns."""
    if isinstance(x, str):
        return '{}'.format(x)
    else:
        return '{:25.5f}'.format(x)


def get_optim_from_econ(x_all_econ, bounds):
    """This function transforms the economic parameters to their optimization counterparts."""
    x_all_optim = []
    for i, value in enumerate(x_all_econ):
        x_all_optim.append(transform_constraint_to_unconstraint(value, bounds[i]))

    # Now we need to replace the information about the flattened covariance matrix with the
    # flattened Cholesky factors.
    shocks_coeffs = x_all_econ[43:53]
    for i in [0, 4, 7, 9]:
        shocks_coeffs[i] **= 2

    shocks = np.zeros((4, 4))
    shocks[0, :] = shocks_coeffs[0:4]
    shocks[1, 1:] = shocks_coeffs[4:7]
    shocks[2, 2:] = shocks_coeffs[7:9]
    shocks[3, 3:] = shocks_coeffs[9:10]

    shocks_cov = shocks + shocks.T - np.diag(shocks.diagonal())
    shocks_cholesky = np.linalg.cholesky(shocks_cov)

    x_all_optim[43:53] = shocks_cholesky[np.tril_indices(4)]

    return np.array(x_all_optim)


def get_econ_from_optim(x_all_optim, bounds):
    """This function transforms the optimization parameters to their economic counterparts."""
    x_all_econ = []
    for i, value in enumerate(x_all_optim):
        x_all_econ.append(transform_unconstraint_to_constraint(value, bounds[i]))

    # Now we need to transform the Cholesky matrix to the flattened covariance matrix.
    x_all_econ[43:53] = cholesky_to_coeffs(extract_cholesky(x_all_optim)[0])

    return np.array(x_all_econ)


def transform_unconstraint_to_constraint(value_unconstraint, bounds):
    """This function accounts for the bounds that might be imposed by the economics of the
    underlying parameter or the user directly."""
    # We first deal with the case where there are neither upper or lower bounds defined.
    if all(x is None for x in bounds):
        return value_unconstraint

    lower, upper = bounds
    if upper is None and lower is not None:
        value_constraint = lower + np.exp(value_unconstraint)

    elif upper is not None and lower is None:
        value_constraint = upper - np.exp(value_unconstraint)

    else:
        interval = upper - lower
        value_constraint = lower + interval / (1.0 + np.exp(-value_unconstraint))

    return value_constraint


def transform_constraint_to_unconstraint(value_constraint, bounds):
    """This function accounts undoes the bounds that were imposed for the economic parameters."""
    # We first deal with the case where there are neither upper or lower bounds defined.
    if all(x is None for x in bounds):
        return value_constraint

    lower, upper = bounds

    if lower is not None and upper is None:
        value_unconstraint = np.log(value_constraint - lower)

    elif lower is None and upper is not None:
        value_unconstraint = np.log(upper - value_constraint)
    else:
        interval = upper - lower
        transform = (value_constraint - lower) / interval

        value_unconstraint = np.log(transform / (1.0 - transform))

    return value_unconstraint


def get_scales(x_free_optim_unscaled):
    """This function calculated the scaling factors based on the starting values."""
    # Initialize container
    scales = []

    for i, x in enumerate(x_free_optim_unscaled):
        if x == 0.0:
            scale = 1
        else:
            magnitude = int(floor(log10(abs(x))))
            if magnitude == 0:
                scale = 1.0 / 10.0
            else:
                scale = (10 ** magnitude) ** (-1) / 10.0
        scales.append(scale)

    return np.array(scales)


def apply_scaling(x_free_optim, scales, request):
    """This function applies the scales to the parameter vector."""
    if request == 'do':
        out = np.multiply(x_free_optim, scales)
    elif request == 'undo':
        out = np.multiply(x_free_optim, scales ** (-1))
    else:
        raise AssertionError

    return out


def moments_dict_to_list(moments_dict):
    """This function constructs a list of available moments based on the moment dictionary."""
    moments_list = []
    for group in ['Choice Probability', 'Wage Distribution']:
        for period in sorted(moments_dict[group].keys()):
            moments_list.extend(moments_dict[group][period])
    return moments_list


def get_starting_values(fname):
    """This function returns the starting values from the class instance."""
    respy_base = respy.RespyCls(fname)
    optim_paras, num_paras = dist_class_attributes(respy_base, 'optim_paras', 'num_paras')

    x_all_econ = get_optim_paras(optim_paras, num_paras, 'all', True)
    x_all_econ[43:53] = cholesky_to_coeffs(extract_cholesky(x_all_econ)[0])
    x_all_optim_start = get_optim_from_econ(x_all_econ, optim_paras['paras_bounds'])

    x_free_optim_start = []
    for i, value in enumerate(x_all_optim_start):
        if optim_paras['paras_fixed'][i]:
            continue
        x_free_optim_start.append(value)

    scales = get_scales(x_free_optim_start)
    x_free_optim_start_scaled = apply_scaling(x_free_optim_start, scales, 'do')

    return x_free_optim_start_scaled, scales


def get_starting_values_econ(fname):
    """This function returns the starting values from the class instance."""
    respy_base = respy.RespyCls(fname)
    optim_paras, num_paras = dist_class_attributes(respy_base, 'optim_paras', 'num_paras')

    x_all_econ = get_optim_paras(optim_paras, num_paras, 'all', True)
    x_all_econ[43:53] = cholesky_to_coeffs(extract_cholesky(x_all_econ)[0])

    # TODO: This is only needed because of the crazy RESPY setup.
    paras_fixed_reordered = optim_paras['paras_fixed'].copy()
    paras_fixed = paras_fixed_reordered[:]
    for old, new in PARAS_MAPPING:
        paras_fixed[old] = paras_fixed_reordered[new]

    x_free_econ_start = list()
    for i, value in enumerate(x_all_econ):
        if paras_fixed[i]:
            continue
        x_free_econ_start.append(value)

    return x_free_econ_start


def store_information(label):
    """This function stores the information about each meaningful event during the optimization."""
    for fname in glob.glob('smm_estimation/*.respy.*'):
        shutil.copy(fname, fname.replace('current.respy', label))


def get_random_string(length=5):
    """This function generates a random string."""
    str_ = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return str_



def smm_sample_pyth(state_space_info, disturbances, respy_obj):
    """This function is a wrapper that is supposed to facilitate the application of SMM
    estimation for the RESPY package."""
    states_all, states_number_period, mapping_state_idx, max_states_period = state_space_info
    periods_draws_emax, periods_draws_sims = disturbances

    num_periods, edu_spec, optim_paras, num_draws_emax, is_debug, is_interpolated, \
    num_points_interp, is_myopic, num_agents_sim, num_agents_est, seed_sim, file_sim, num_types, \
    is_myopic = dist_class_attributes(respy_obj, 'num_periods', 'edu_spec', 'optim_paras',
        'num_draws_emax', 'is_debug', 'is_interpolated', 'num_points_interp', 'is_myopic',
        'num_agents_sim', 'num_agents_est', 'seed_sim', 'file_sim', 'num_types', 'is_myopic')

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
    file_sim, num_paras, num_types, num_agents_est, is_attach = dist_class_attributes(respy_obj,
        'optim_paras', 'num_periods', 'edu_spec', 'is_debug', 'num_draws_emax', 'seed_emax',
        'is_interpolated', 'num_points_interp', 'is_myopic', 'tau', 'num_procs', 'num_agents_sim',
        'num_draws_prob', 'seed_prob', 'seed_sim', 'optimizer_options',  'optimizer_used',
        'maxfun', 'precond_spec', 'file_sim', 'num_paras', 'num_types', 'num_agents_est',
        'is_attach')

    args = (optim_paras, is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_spec, is_debug, num_draws_prob, num_agents_sim, seed_prob, seed_emax,
        tau, num_procs, 'simulate', seed_sim, optimizer_options, optimizer_used, maxfun, num_paras,
        precond_spec, file_sim, None, num_types, num_agents_est, is_attach)

    write_resfort_initialization(*args)

    test = ROOT_DIR + '/.bld/fortran/resfort_slave'
    worker = MPI.COMM_SELF.Spawn(test, maxprocs=num_procs - 1)

    return worker


class AggregatorCls(object):

    def __init__(self):
        """Constructor for the aggregation manager."""
        self.attr = dict()
        self.attr['fun_step'] = None
        self.attr['num_step'] = 0
        self.attr['num_eval'] = 0

    def run(self, e):
        """This function constantly checks for the best evaluation point."""
        fname = 'blackbox_best/smm_estimation/smm_monitoring.pkl'
        self.attr['fun_step'] = pkl.load(open(fname, 'rb'))['Current'].ix[0]

        # We need to set up the logging with the baseline information from the starting values.
        fmt_record = ' {:>25}{:>25.5f}{:>25.5f}{:>25}{:>25}\n'

        with open('blackbox.respy.log', 'w') as outfile:
            fmt_ = ' {:>25}{:>25}{:>25}{:>25}{:>25}\n\n'
            outfile.write(fmt_.format(*['Time', 'Criterion', 'Best', 'Evaluation', 'Step']))
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = list()
            line += [time, self.attr['fun_step'], self.attr['fun_step']]
            line += [self.attr['num_eval'], self.attr['num_step']]
            outfile.write(fmt_record.format(*line))

        self.attr['num_eval'] += 1

        # This setup ensures that a complete check of all directories is done after the algorithm
        # concludes. Otherwise, this might be terminated in an intermediate step.
        while not e.is_set():
            sleep(2)
            self._collect_information()
        else:
            self._collect_information()

    def _collect_information(self):
        """This method iterates over the BLACKBOX directories."""
        fmt_record = ' {:>25}{:>25.5f}{:>25.5f}{:>25}{:>25}\n'

        dirnames = glob.glob('blackbox_*')

        for dirname in dirnames:

            if dirname in ['blackbox_best']:
                continue

            if os.path.exists(dirname + '/.ready.blackbox.info'):
                fname = dirname + '/smm_estimation/smm_monitoring.pkl'
                candidate = pkl.load(open(fname, 'rb'))['Current'].ix[0]
                if candidate < self.attr['fun_step']:
                    shutil.rmtree('blackbox_best')
                    shutil.copytree(dirname, 'blackbox_best')
                    self.attr['fun_step'] = candidate
                    self.attr['num_step'] += 1

                shutil.rmtree(dirname, ignore_errors=True)

                # Update real-time logging
                with open('blackbox.respy.log', 'a') as outfile:
                    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    line = []
                    line += [time, candidate, self.attr['fun_step']]
                    line += [self.attr['num_eval'], self.attr['num_step']]
                    outfile.write(fmt_record.format(*line))
                self.attr['num_eval'] += 1


def blackbox_wrapper(est_obj, x_free_econ_eval):
    """This function serves as a simple wrapper that allows to use the BLACKBOX optimization
    algorithm"""
    name = 'blackbox_' + str(uuid.uuid4())

    os.mkdir(name)
    os.chdir(name)

    func = est_obj.criterion(True, x_free_econ_eval)

    os.chdir('../')

    os.mknod(name + '/.ready.blackbox.info')

    return func


def cleanup(is_start=False):
    """This module cleans the BLACKBOX directories."""
    for dirname in glob.glob('blackbox_*'):
        if not is_start and 'best' in dirname:
            continue
        shutil.rmtree(dirname)

    fname = 'blackbox.respy.csv'
    if is_start and os.path.exists(fname):
        os.remove(fname)

    fname = '.est_obj.blackbox.pkl'
    if os.path.exists(fname):
        os.remove(fname)

