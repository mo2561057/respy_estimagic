from functools import partial

import numpy as np

from respy.python.shared.shared_auxiliary import dist_class_attributes, get_num_obs_agent
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import MISSING_INT, HUGE_FLOAT
import f2py_interface as respy_f2py

from respy_smm.auxiliary_depreciation import respy_obj_from_new_init
from respy_smm.auxiliary import is_valid_covariance_matrix, get_processed_dataset
from respy_smm.auxiliary_depreciation import x_all_econ_new_to_old
from respy_smm.clsEstimation import EstimationCls
from respy_smm.clsLogging import logger_obj
from respy_smm.src import smm_interface
from respy_smm import HUGE_INT


class MaximumLikelihoodEstimationCls(EstimationCls):
    """This class manages the distribution of the use requests throughout the toolbox."""
    def __init__(self, init_file, max_evals=HUGE_INT):

        super().__init__()

        self.data_array = get_processed_dataset(init_file)
        self.respy_base = respy_obj_from_new_init(init_file)
        self.max_evals = max_evals

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
            self.check_termination()
            return HUGE_FLOAT

        fval = self.criterion_function(x_all_econ_new_to_old(x_all_econ_current))
        np.testing.assert_equal(np.isnan(fval), False)

        self.wrapping_up_evaluation(x_all_econ_current, fval)
        self.check_termination()

        return fval

    def _prepare_evaluate(self):
        """This method runs a host of preparations for the evaluation of the criterion function
        that are specific to the current RESPY implementation. So, major cleanup as we settle on
        the revised interface."""
        labels = list()
        labels += ['num_procs', 'num_periods', 'is_debug', 'seed_emax', 'seed_sim']
        labels += ['num_draws_emax', 'num_agents_sim', 'num_types', 'edu_spec', 'optim_paras']
        labels += ['is_interpolated', 'num_points_interp', 'is_myopic', "num_paras", 'tau']
        labels += ['num_draws_prob', 'num_agents_est', 'seed_prob']

        num_procs, num_periods, is_debug, seed_emax, seed_sim, num_draws_emax, num_agents_sim, \
        num_types, edu_spec, optim_paras, is_interpolated, num_points_interp, is_myopic, \
        num_paras, tau, num_draws_prob, num_agents_est, seed_prob = \
            dist_class_attributes(self.respy_base, *labels)

        num_obs_agent = get_num_obs_agent(self.data_array, num_agents_est)

        args = list()
        args += [num_periods, num_types, edu_spec['start'], edu_spec['max']]
        args += [edu_spec['max'] + 1]

        states_all, states_number_period, mapping_state_idx, max_states_period = \
            respy_f2py.wrapper_create_state_space(*args)
        states_all = states_all[:, :max_states_period, :]

        periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax, is_debug)
        periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob, is_debug)

        if self.mpi_setup == MISSING_INT:
            slavecomm = self.mpi_setup
        else:
            slavecomm = self.mpi_setup.py2f()

            from mpi4py import MPI
            for i in range(num_periods):
                for j in range(num_draws_emax):
                    self.mpi_setup.Bcast([periods_draws_emax[i, j, :], MPI.DOUBLE], root=MPI.ROOT)

            for i in range(num_periods):
                for j in range(num_draws_prob):
                    self.mpi_setup.Bcast([periods_draws_prob[i, j, :], MPI.DOUBLE], root=MPI.ROOT)

            data = np.ascontiguousarray(self.data_array, np.float64)
            for i in range(data.shape[0]):
                self.mpi_setup.Bcast([data[i, :], MPI.DOUBLE], root=MPI.ROOT)

        args = list()
        args += [is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic]
        args += [is_debug, self.data_array, num_draws_prob, tau, periods_draws_emax]
        args += [periods_draws_prob, states_all, states_number_period, mapping_state_idx]
        args += [max_states_period, num_agents_est, num_obs_agent, num_types]
        args += [edu_spec['start'], edu_spec['max'], edu_spec['share']]
        args += [num_paras, slavecomm]

        self.criterion_function = partial(smm_interface.wrapper_criterion, *args)
