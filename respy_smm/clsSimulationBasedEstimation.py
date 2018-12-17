"""This module contains the main class that allows for the SMM estimation."""
from functools import partial
import time
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

from respy.python.shared.shared_auxiliary import coeffs_to_cholesky
from respy_smm.auxiliary_depreciation import shocks_spec_new_to_old
from respy_smm.auxiliary_depreciation import respy_spec_old_to_new
from respy_smm.auxiliary import get_communicator
from respy_smm.auxiliary import smm_sample_pyth
from respy_smm.auxiliary import smm_sample_f2py
from respy_smm.clsLogging import logger_obj
from respy_smm.moments import get_moments
from respy_smm import HUGE_FLOAT

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_BUILD
from respy.python.shared.shared_constants import DATA_FORMATS_SIM
from respy.python.shared.shared_constants import DATA_LABELS_SIM
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import MISSING_INT
import respy

sys.path.insert(0, TEST_RESOURCES_BUILD)
import f2py_interface as respy_f2py


class SimulationBasedEstimationCls(object):
    """This class manages the distribution of the use requests throughout the toolbox."""
    def __init__(self, init_file, moments_obs, weighing_matrix, max_evals=None):

        respy_base = respy.RespyCls(init_file)

        optim_paras, num_procs, num_periods, is_debug, seed_emax, seed_sim, \
        num_draws_emax, num_agents_sim, num_types, edu_spec, version = dist_class_attributes(
            respy_base, 'optim_paras', 'num_procs', 'num_periods', 'is_debug', 'seed_emax',
            'seed_sim', 'num_draws_emax', 'num_agents_sim', 'num_types', 'edu_spec',
            'version')

        if num_procs > 1:
            assert 'PMI_SIZE' in os.environ.keys()
            worker = get_communicator(respy_base)
        else:
            worker = MISSING_INT

        self.simulate_sample = None

        self.attr = dict()
        self.attr['mpi_setup'] = worker

        self.attr['x_all_econ_start'] = respy_spec_old_to_new(optim_paras)
        self.attr['paras_free'] = ~np.array(optim_paras['paras_fixed'])
        self.attr['weighing_matrix'] = weighing_matrix
        self.attr['moments_obs'] = moments_obs
        self.attr['num_periods'] = num_periods
        self.attr['respy_base'] = respy_base
        self.attr['max_evals'] = max_evals
        self.attr['num_evals'] = 0
        self.attr['func'] = None

        args = [weighing_matrix, max_evals, optim_paras['paras_fixed']]
        logger_obj.setup_information(*args)

    def criterion(self, x_input):
        """This function evaluates the criterion function for a candidate parametrization."""
        # Distribute class attributes
        x_all_econ_start = self.attr['x_all_econ_start']
        weighing_matrix = self.attr['weighing_matrix']
        num_periods = self.attr['num_periods']
        moments_obs = self.attr['moments_obs']
        paras_free = self.attr['paras_free']
        respy_base = self.attr['respy_base']

        x_all_econ_eval = x_all_econ_start.copy()
        x_all_econ_eval[paras_free] = x_input

        # We are now simulating a sample based on the updated parametrization.
        start = time.time()

        respy_smm = copy.deepcopy(respy_base)

        x_all_econ_eval_respy_old = x_all_econ_eval.copy()
        x_all_econ_eval_respy_old[43:53] = shocks_spec_new_to_old(x_all_econ_eval[43:53])

        # TODO: Move to a proper testing setup once settled on interface.
        arg_1 = x_all_econ_eval[43:53][:4]
        arg_2 = x_all_econ_eval_respy_old[43:53][[0, 4, 7, 9]]
        np.testing.assert_almost_equal(arg_1, arg_2)

        # TODO: This is crude way to ensure positive semidefinite matrix for a valid evaluation.
        #  Now we simply return a huge value.
        try:
            coeffs_to_cholesky(x_all_econ_eval_respy_old[43:53])
        except np.linalg.linalg.LinAlgError:
            msg = 'invalid evaluation due to lack of proper covariance matrix'
            logger_obj.record_abort_eval(msg)
            return HUGE_FLOAT

        respy_smm.update_optim_paras(x_all_econ_eval_respy_old)
        df_smm = self.create_smm_sample(respy_smm)
        moments_sim = self._get_sim_moments(df_smm)

        stop = time.time()
        duration = int(stop - start)

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
            msg = 'invalid evaluation due to missing moments'
            logger_obj.record_abort_eval(msg)
            return HUGE_FLOAT

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

        args = [[func, mad] + x_all_econ_eval.tolist(), stats_obs, stats_sim, respy_smm, duration]
        logger_obj.record(*args)

        self.attr['func'] = func
        self._check_termination()

        return func

    def create_smm_sample(self, respy_obj):
        """This method creates a dataframe for the evaluation of the criterion function."""
        # We need to incur the proper setup cost.
        if self.simulate_sample is None:

            worker = self.attr['mpi_setup']

            num_procs, num_periods, is_debug, seed_emax, seed_sim, num_draws_emax, num_agents_sim,\
            num_types, edu_spec, version = dist_class_attributes(respy_obj, 'num_procs',
                'num_periods', 'is_debug', 'seed_emax', 'seed_sim', 'num_draws_emax',
                'num_agents_sim', 'num_types', 'edu_spec', 'version')

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

    def get_attr(self, key):
        """Get attributes."""
        return self.attr[key]

    def _check_termination(self):
        """This method ensures a gentle shutdown."""
        self.attr['num_evals'] += 1

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

    @staticmethod
    def _get_sim_moments(df):
        """This function computes the moments from a dataset."""
        return get_moments(df)
