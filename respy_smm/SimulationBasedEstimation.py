"""This module contains the main class that allows for the SMM estimation."""
from functools import partial
import os

import pandas as pd
import numpy as np

from respy_smm.auxiliary_depreciation import shocks_spec_new_to_old
from respy_smm.auxiliary_depreciation import respy_obj_from_new_init
from respy_smm.auxiliary import is_valid_covariance_matrix
from respy_smm.clsEstimation import EstimationCls
from respy_smm.auxiliary import smm_sample_f2py
from respy_smm.clsLogging import logger_obj
from respy_smm import HUGE_FLOAT
from respy_smm import HUGE_INT

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import DATA_FORMATS_SIM
from respy.python.shared.shared_constants import DATA_LABELS_SIM
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import MISSING_INT
import f2py_interface as respy_f2py


class SimulationBasedEstimationCls(EstimationCls):
    """This class manages the distribution of the use requests throughout the toolbox."""
    def __init__(self, init_file, moments_obs, weighing_matrix, get_moments, max_evals=HUGE_INT):

        super().__init__()

        self.respy_base = respy_obj_from_new_init(init_file)
        self.weighing_matrix = weighing_matrix
        self.get_moments = get_moments
        self.moments_obs = moments_obs
        self.max_evals = max_evals

        self.simulate_sample = None

        self.set_derived_attributes()

        # We always lock in an evaluation at the starting values and prepare for the evaluation
        # of the criterion function.
        self._prepare_evaluate()
        self.evaluate(self.x_free_econ_start)

    def evaluate(self, x_free_econ):
        """This method evaluates the criterion function for a candidate parametrization proposed
        by the optimizer."""
        x_all_econ_current = self.construct_complete_parameters(x_free_econ)

        if not is_valid_covariance_matrix(x_all_econ_current[43:53]):
            msg = 'invalid evaluation due to lack of proper covariance matrix'
            logger_obj.record_abort_eval(msg)
            return HUGE_FLOAT

        # We now move into the territory of the code which will be simplified as we improve the
        # RESPY implementation.
        x_all_econ_eval_respy_old = x_all_econ_current.copy()
        x_all_econ_eval_respy_old[43:53] = shocks_spec_new_to_old(x_all_econ_current[43:53])

        self.respy_base.update_optim_paras(x_all_econ_eval_respy_old)
        mat_smm = self.simulate_sample(self.respy_base)

        df_smm = pd.DataFrame(replace_missing_values(mat_smm), columns=DATA_LABELS_SIM)
        df_smm = df_smm.astype(DATA_FORMATS_SIM)
        df_smm.set_index(['Identifier', 'Period'], drop=False, inplace=True)
        moments_sim = self.get_moments(df_smm)

        # TODO: Now we move all moments from a dictionary to an array. Is there a way to avoid
        #  this transformation? This is not nicely done at this point, but I cannot think of a
        #  nice way that requires a large OOP overhead. I want to make sure that the moments can
        #  be provided in the simple form of a function.
        stats_obs, stats_sim = [], []
        for group in self.moments_obs.keys():
            for period in range(max(self.moments_obs[group].keys()) + 1):
                if period not in moments_sim[group].keys():
                    continue
                if period not in self.moments_obs[group].keys():
                    continue
                stats_obs.extend(self.moments_obs[group][period])
                stats_sim.extend(moments_sim[group][period])

        # We need to deal with the special case where it might happen that some moments for the
        # wage distribution are available in the observed dataset but not the simulated dataset.
        is_valid = len(stats_obs) == len(stats_sim) == len(np.diag(self.weighing_matrix))
        if is_valid:
            stats_diff = np.array(stats_obs) - np.array(stats_sim)
            fval = float(np.dot(np.dot(stats_diff, self.weighing_matrix), stats_diff))
        else:
            msg = 'invalid evaluation due to missing moments'
            logger_obj.record_abort_eval(msg)
            self.check_termination()
            fval = HUGE_FLOAT

        self.wrapping_up_evaluation(x_all_econ_current, fval)
        self._logging_smm(stats_obs, stats_sim)
        self.check_termination()

        return fval

    def _prepare_evaluate(self):
        """This method runs a host of preparations for the evaluation of the criterion function
        that are specific to the current RESPY implementation. So, major cleanup as we settle on
        the revised interface."""
        labels = list()
        labels += ['num_procs', 'num_periods', 'is_debug', 'seed_emax', 'seed_sim']
        labels += ['num_draws_emax', 'num_agents_sim', 'num_types', 'edu_spec', 'version']

        num_procs, num_periods, is_debug, seed_emax, seed_sim, num_draws_emax, num_agents_sim, \
        num_types, edu_spec, version = dist_class_attributes(self.respy_base, *labels)

        periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax, is_debug)
        periods_draws_sims = create_draws(num_periods, num_agents_sim, seed_sim, is_debug)

        disturbances = (periods_draws_emax, periods_draws_sims)

        # We want to maintain a pure PYTHON version for testing purposes.
        args = list()
        args += [num_periods, num_types, edu_spec['start'], edu_spec['max'], edu_spec['max'] + 1]
        state_space_info = respy_f2py.wrapper_create_state_space(*args)
        if self.mpi_setup == MISSING_INT:
            slavecomm = self.mpi_setup
        else:
            slavecomm = self.mpi_setup.py2f()

        self.simulate_sample = partial(smm_sample_f2py, state_space_info, disturbances, slavecomm)

    def _logging_smm(self, stats_obs, stats_sim):
        """This method contains logging capabilities that are just relevant for the SMM routine."""
        fname = 'monitoring.estimagic.smm.info'
        if self.num_evals and os.path.exists(fname):
            os.unlink(fname)

        with open(fname, 'a+') as outfile:

            fmt_ = '\n\n{:>8}{:>15}\n\n'
            outfile.write(fmt_.format('EVALUATION', self.num_evals))

            fmt_ = '{:>8}' + '{:>15}' * 4 + '\n\n'
            info = ['Moment', 'Observed', 'Simulated', 'Difference', 'Weight']
            outfile.write(fmt_.format(*info))

            for i, moment in enumerate(stats_obs):

                stat_obs, stat_sim = stats_obs[i], stats_sim[i]
                info = [i, stat_obs, stat_sim, abs(stat_obs - stat_sim), self.weighing_matrix[i, i]]

                fmt_ = '{:>8}' + '{:15.5f}' * 4 + '\n'
                outfile.write(fmt_.format(*info))
