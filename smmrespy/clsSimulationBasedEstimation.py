"""This module contains the main class that allows for the SMM estimation."""
from functools import partial
import warnings
import copy
import sys
import os

import pandas as pd
import numpy as np

if 'PMI_SIZE' in os.environ.keys():
    try:
        from mpi4py import MPI
    except ImportError:
        pass

from smmrespy.auxiliary import get_optim_from_econ
from smmrespy.auxiliary import get_econ_from_optim
from smmrespy.auxiliary import get_communicator
from smmrespy.auxiliary import smm_sample_pyth
from smmrespy.auxiliary import smm_sample_f2py
from smmrespy.auxiliary import apply_scaling
from smmrespy.auxiliary import format_column
from smmrespy.moments import get_moments
from smmrespy.config_smmrespy import HUGE_FLOAT

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_BUILD
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_constants import DATA_FORMATS_SIM
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.python.shared.shared_constants import DATA_LABELS_SIM
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import MISSING_INT
from respy.clsRespy import PARAS_MAPPING
import respy

sys.path.insert(0, TEST_RESOURCES_BUILD)
import f2py_interface as respy_f2py


class SimulationBasedEstimationCls(object):
    """This class manages the distribution of the use requests throughout the toolbox."""
    def __init__(self, init_file, moments_obs, weighing_matrix, scales=None, max_evals=None):

        respy_base = respy.RespyCls(init_file)

        num_paras, optim_paras, num_procs, num_periods, is_debug, seed_emax, seed_sim, \
        num_draws_emax, num_agents_sim, num_types, edu_spec, version = dist_class_attributes(
            respy_base, 'num_paras', 'optim_paras', 'num_procs', 'num_periods', 'is_debug',
            'seed_emax', 'seed_sim', 'num_draws_emax', 'num_agents_sim', 'num_types', 'edu_spec',
            'version')

        if num_procs > 1:
            assert 'PMI_SIZE' in os.environ.keys()
            worker = get_communicator(respy_base)
        else:
            worker = MISSING_INT

        self.simulate_sample = None

        self.attr = dict()
        self.attr['mpi_setup'] = worker

        self.attr['paras_bounds'] = optim_paras['paras_bounds']
        self.attr['paras_fixed'] = optim_paras['paras_fixed']
        self.attr['weighing_matrix'] = weighing_matrix
        self.attr['moments_obs'] = moments_obs
        self.attr['respy_base'] = respy_base
        self.attr['num_paras'] = num_paras
        self.attr['max_evals'] = max_evals
        self.attr['num_periods'] = num_periods

        self.attr['df_info'] = None
        self.attr['num_evals'] = 0
        self.attr['num_steps'] = 0
        self.attr['func'] = None

        if scales is None:
            num_free = optim_paras['paras_fixed'].count(False)
            scales = np.tile(1.0, num_free)
        else:
            pass
        self.attr['scales'] = scales

        # We need to construct sound vectors that are either all economic or all optimizer
        # parameter values. This is not nicely done with the RESPY routines at all.
        x_all_econ = get_optim_paras(optim_paras, num_paras, 'all', True)
        x_all_econ[43:53] = cholesky_to_coeffs(extract_cholesky(x_all_econ)[0])
        self.attr['x_all_econ_start'] = x_all_econ.copy()
        self.attr['x_all_optim_start'] = get_optim_from_econ(x_all_econ, self.attr['paras_bounds'])

    def create_smm_sample(self, respy_obj):
        """This method creates a dataframe for the ..."""
        # We need to incur the proper setup cost.
        if self.simulate_sample is None:

            worker = self.attr['mpi_setup']

            num_paras, optim_paras, num_procs, num_periods, is_debug, seed_emax, seed_sim, \
            num_draws_emax, num_agents_sim, num_types, edu_spec, version = dist_class_attributes(
                respy_obj, 'num_paras', 'optim_paras', 'num_procs', 'num_periods', 'is_debug',
                'seed_emax', 'seed_sim', 'num_draws_emax', 'num_agents_sim', 'num_types',
                'edu_spec',
                'version')

            periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax, is_debug)
            periods_draws_sims = create_draws(num_periods, num_agents_sim, seed_sim, is_debug)

            disturbances = (periods_draws_emax, periods_draws_sims)

            # We want to maintain a pure PYTHON version for testing purposes.
            if version in ['FORTRAN']:
                args = []
                args += [num_periods, num_types, edu_spec['start'], edu_spec['max']]
                args += [edu_spec['max'] + 1]
                state_space_info = respy_f2py.wrapper_create_state_space(*args)
                if worker == MISSING_INT:
                    slavecomm = worker
                else:
                    slavecomm = worker.py2f()
                func = partial(smm_sample_f2py, state_space_info, disturbances, slavecomm)
            elif version in ['PYTHON']:
                args = [num_periods, num_types, edu_spec]
                state_space_info = pyth_create_state_space(*args)
                func = partial(smm_sample_pyth, state_space_info, disturbances)
            else:
                raise NotImplementedError
            self.simulate_sample = func

        mat_smm = self.simulate_sample(respy_obj)

        data_frame = pd.DataFrame(replace_missing_values(mat_smm), columns=DATA_LABELS_SIM)
        data_frame = data_frame.astype(DATA_FORMATS_SIM)
        data_frame.set_index(['Identifier', 'Period'], drop=False, inplace=True)

        return data_frame

    def criterion(self, is_econ, x_input):
        """This function evaluates the criterion function for a candidate parametrization."""
        # Distribute class attributes
        x_all_optim_start = self.attr['x_all_optim_start']
        weighing_matrix = self.attr['weighing_matrix']
        paras_bounds = self.attr['paras_bounds']
        paras_fixed = self.attr['paras_fixed']
        moments_obs = self.attr['moments_obs']
        respy_base = self.attr['respy_base']
        num_paras = self.attr['num_paras']
        scales = self.attr['scales']
        num_periods = self.attr['num_periods']

        # We first need to undo the scaling exercise.
        if not is_econ:
            x_free_optim = apply_scaling(x_input, scales, 'undo')

            # Extend optimization parameter to include all values.
            x_all_optim_eval = x_all_optim_start.copy()
            j = 0
            for i in range(num_paras):
                if paras_fixed[i]:
                    continue
                x_all_optim_eval[i] = x_free_optim[j]
                j += 1

            x_all_econ_eval = get_econ_from_optim(x_all_optim_eval, paras_bounds)
        else:

            # TODO: This is only needed because of the crazy RESPY setup.
            # TODO: When using the economic parameters then there is not scaling done at all.
            x_all_econ_eval = self.attr['x_all_econ_start'].copy()
            paras_fixed_reordered = self.attr['paras_fixed'].copy()
            paras_fixed = paras_fixed_reordered[:]
            for old, new in PARAS_MAPPING:
                paras_fixed[old] = paras_fixed_reordered[new]

            j = 0
            for i in range(num_paras):
                if paras_fixed[i]:
                    continue
                x_all_econ_eval[i] = x_input[j]
                j += 1

        # We are now simulating a sample based on the updated parameterization.
        respy_smm = copy.deepcopy(respy_base)
        respy_smm.update_optim_paras(x_all_econ_eval)
        df_smm = self.create_smm_sample(respy_smm)

        # Construct moments for simulated dataset and evaluate the criterion function.
        moments_sim = self._get_sim_moments(df_smm)

        stats_obs, stats_sim = [], []
        for group in ['Choice Probability', 'Wage Distribution']:
            for period in range(num_periods):

                if period not in moments_sim[group].keys():
                    continue
                if period not in moments_obs[group].keys():
                    continue

                stats_obs.extend(moments_obs[group][period])
                stats_sim.extend(moments_sim[group][period])

        # The calculation of the criterion function fails if the not all moments that were
        # calculated on the observed dataset are also available for the simulated dataset. This
        # can happen, for example, if nobody in the simulated dataset work in a particular period
        # and thus no information on the wage distribution is available.
        stats_diff = np.array(stats_obs) - np.array(stats_sim)
        try:
            func = float(np.dot(np.dot(stats_diff, weighing_matrix), stats_diff))
        except ValueError:
            warnings.warn('invalid evaluation as not all moments available')
            func = HUGE_FLOAT

        # We also want to log the choice probabilities for logging purposes.
        probs_obs, probs_sim = [], []
        for group in ['Choice Probability']:
            for period in range(num_periods):
                if period not in moments_sim[group].keys():
                    continue
                if period not in moments_obs[group].keys():
                    continue
                probs_obs.extend(moments_obs[group][period])
                probs_sim.extend(moments_sim[group][period])

        mad = float(np.mean(np.abs(np.array(probs_obs) - np.array(probs_sim))))

        # This ensures that the optimizer does not get stuck in extreme areas where most of the
        # non-wage cells are zero. The zero choice probabilities for non-wage choices do not
        # result in ValueError above, that is why this is handled here explicitly.
        is_invalid = False
        for i, label in enumerate(['Wage A', 'Wage B', 'School', 'Home']):
            stats_check_obs, stats_check_sim = list(), list()
            for period in range(num_periods):
                if period not in moments_sim[group].keys():
                    continue
                if period not in moments_obs[group].keys():
                    continue

                stats_check_obs.append(moments_obs['Choice Probability'][period][i])
                stats_check_sim.append(moments_sim['Choice Probability'][period][i])

            num_obs = (np.array(stats_check_obs) > 0.05).sum()
            num_sim = (np.array(stats_check_sim) > 0.05).sum()

            if num_sim < 0.5 * num_obs:
                is_invalid = True

        if is_invalid:
            warnings.warn('invalid evaluation as there are not enough thick cells')
            func = HUGE_FLOAT

        args = [func, mad] + x_all_econ_eval.tolist(), stats_obs, stats_sim, weighing_matrix, respy_smm
        self._logging(*args)
        self.attr['func'] = func

        return func

    @staticmethod
    def _get_sim_moments(df):
        """This function computes the moments from a dataset."""
        return get_moments(df)

    def _logging(self, info_update, stats_obs, stats_sim, weighing_matrix, respy_smm):
        """This method logs the progress of the estimation."""
        # Distribute class attributes
        num_paras = self.attr['num_paras']
        df_info = self.attr['df_info']

        self.attr['func'] = info_update[0]

        # Initialize empty canvas if required
        if df_info is None:
            df_info = dict()
            df_info['Label'] = ['Criterion', 'MAD'] + [str(i) for i in range(num_paras)]
            for label in ['Start', 'Step', 'Current']:
                df_info[label] = info_update
            df_info = pd.DataFrame(df_info)

        # We set up some basic logging for monitoring.
        is_step = df_info['Step'][0] >= info_update[0]

        df_info['Current'] = info_update
        self.attr['num_evals'] += 1

        if is_step:
            respy_smm.write_out('smm_monitoring.step.ini')
            df_info['Step'] = info_update
            self.attr['num_steps'] += 1

        formatters = dict()
        columns = ['Label', 'Start', 'Step', 'Current']
        for label in columns:
            formatters[label] = format_column

        fname = 'smm_monitoring'
        df_info.to_string(open(fname + '.info', 'w'), index=False, columns=columns,
                          justify='justify-all', formatters=formatters)
        df_info.to_pickle(fname + '.pkl')
        self.attr['df_info'] = df_info

        # We also want to provide some information on each step of the optimizer.
        fname = 'smm_monitoring.log'
        if self.attr['num_evals'] == 1:
            open(fname, 'w').close()

        # TODO: Deal wit huge value of not all moments identifier
        with open(fname, 'a') as outfile:
            fmt_ = '    {:<25}\n\n'
            outfile.write(fmt_.format(*['OVERVIEW']))
            fmt_ = ' {:>25}{:>25}{:>25}\n\n'
            outfile.write(fmt_.format(*['Evaluation', 'Step', 'Criterion']))
            fmt_ = ' {:>25}{:>25}{:>25.5f}\n'

            line = [self.attr["num_evals"], self.attr["num_steps"], self.attr["func"]]
            outfile.write(fmt_.format(*line))
            outfile.write('\n\n')

            # ANd log the value of the free parameters
            fmt_ = '    {:<25}\n\n'
            outfile.write(fmt_.format(*['FREE ECONOMICS PARAMETERS']))

            fmt_ = ' {:>25}{:>25}\n\n'
            line = ['Identifier', 'Value']
            outfile.write(fmt_.format(*line))

            fmt_ = ' {:>25}{:>25.5f}\n'
            count = 0
            for i, value in enumerate(info_update[2:]):
                if not self.attr['paras_fixed'][i]:
                    line = [count, value]
                    outfile.write(fmt_.format(*line))

                    count += 1

            # We want to be able to inspect the moments.
            fmt_ = '    {:<25}\n\n'
            outfile.write(fmt_.format(*['MOMENTS']))

            fmt_ = ' {:>25}{:>25}{:>25}{:>25}{:>25}\n\n'
            labels = ["Identifier", "Observation", "Simulation", 'Difference', 'Weight']
            outfile.write(fmt_.format(*labels))
            fmt_ = ' {:>25}{:25.5f}{:25.5f}{:25.5f}{:25.5f}\n'

            if self.attr['func'] != HUGE_FLOAT:
                for i in range(len(stats_obs)):
                    diff = np.abs(stats_obs[i] - stats_sim[i])
                    line = [i, stats_obs[i], stats_sim[i], diff, weighing_matrix[i, i]]
                    outfile.write(fmt_.format(*line))
                outfile.write('\n\n')

            outfile.write('\n ' + '-' * 125 + '\n\n')

        # We want to keep tight control of the number of evaluations.
        if self.attr['max_evals'] is not None:
            is_termination = self.attr['num_evals'] >= self.attr['max_evals']
            if is_termination:
                try:
                    worker = self.attr['mpi_setup']
                    cmd = np.array(1, dtype='int32')
                    worker.Bcast([cmd, MPI.INT], root=MPI.ROOT)
                except AttributeError:
                    pass
                raise StopIteration

