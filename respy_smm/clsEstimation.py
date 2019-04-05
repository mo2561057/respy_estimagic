from collections import OrderedDict
import pickle as pkl
import copy

import numpy as np

from respy_smm.auxiliary_depreciation import shocks_spec_new_to_old
from respy_smm.auxiliary_depreciation import respy_spec_old_to_new
from respy_smm.auxiliary_depreciation import respy_ini_old_to_new
from respy_smm.auxiliary import get_communicator
from respy_smm.auxiliary import get_mpi

from respy.python.shared.shared_constants import MISSING_INT
from respy.python.shared.shared_constants import HUGE_FLOAT


class EstimationCls(object):

    def __init__(self):

        # We have several attributes that are shared across the children classes. These are
        # declared here for convenience.
        self.x_all_econ = OrderedDict()
        self.criterion_function = None
        self.logging_container = None
        self.x_free_econ_start = None
        self.fval = OrderedDict()
        self.paras_free = None
        self.num_paras = None
        self.mpi_setup = None
        self.is_start = True
        self.num_evals = 0
        self.num_steps = 0

        # We need to set up containers for logging our progress.
        self.info = OrderedDict()
        self.info['x_econ_all'] = list()
        self.info['is_step'] = list()
        self.info['fval'] = list()

    def construct_complete_parameters(self, x_free_econ):
        x_all_econ_current = self.x_all_econ['start'].copy()
        x_all_econ_current[self.paras_free] = x_free_econ
        return x_all_econ_current

    def wrapping_up_evaluation(self, x_all_econ_current, fval):
        self.num_evals += 1

        is_step = self.fval['step'] > fval
        is_current = True

        if self.is_start:
            self.fval['start'] = fval
            self.is_start = False

        if is_current:
            self.x_all_econ['current'] = x_all_econ_current
            self.fval['current'] = fval

        if is_step:
            self.x_all_econ['step'] = x_all_econ_current
            self.fval['step'] = fval
            self.num_steps += 1

        self.monitoring(is_step)

    def check_termination(self):
        is_termination = self.num_evals >= self.max_evals
        if is_termination:
            self.terminate()

    def terminate(self, is_gentle=False):
        if hasattr(self.mpi_setup, 'Bcast'):
            MPI = get_mpi()
            cmd = np.array(1, dtype='int32')
            self.mpi_setup.Bcast([cmd, MPI.INT], root=MPI.ROOT)
        if not is_gentle:
            raise StopIteration

    def monitoring(self, is_step):
        """We provide some basic monitoring during the optimization routine. This is monitoring
        is independent of the particular estimation strategy."""
        self.logging_container[0, :] = list(self.fval.values())
        for i, k in enumerate(self.x_all_econ.keys()):
            self.logging_container[1:, i] = self.x_all_econ[k]

        with open('monitoring.estimagic.info', 'w') as outfile:

            outfile.write('\n Criterion Function\n\n')
            fmt_ = ' {:>25}' * 4 + '\n\n'
            line = ['', 'Start', 'Step', 'Current']
            outfile.write(fmt_.format(*line))

            fmt_ = ' {:>25}' + ' {:25.7f}' * 3 + '\n'
            line = [''] + self.logging_container[0, :].tolist()
            outfile.write(fmt_.format(*line))

            outfile.write('\n Economic Parameters\n\n')

            fmt_ = ' {:>25}' * 4 + '\n\n'
            line = ['Identifier', 'Start', 'Step', 'Current']
            outfile.write(fmt_.format(*line))

            fmt_ = ' {:>25}' + ' {:25.7f}' * 3 + '\n'
            for i in range(self.num_paras):
                line = [i] + self.logging_container[i + 1, :].tolist()
                outfile.write(fmt_.format(*line))

            outfile.write('\n\n')
            outfile.write(' Number of evaluations: {:>25}\n'.format(self.num_evals))
            outfile.write(' Number of steps:       {:>25}\n'.format(self.num_steps))
            outfile.write('\n')

        # We also need a pickle for some more information for monitoring the estimation.
        self.info['x_econ_all'] += [self.x_all_econ['current']]
        self.info['fval'] += [self.fval['current']]
        self.info['is_step'] += [is_step]

        pkl.dump(self.info, open('monitoring.estimagic.pkl', 'wb'))

        # We want to ensure an easy way to restart from the best point of evaluation.
        if is_step:
            x_all_econ_eval_respy_old = self.x_all_econ['current'].copy()
            x_all_econ_eval_respy_old[43:53] = \
                shocks_spec_new_to_old(x_all_econ_eval_respy_old[43:53])

            respy_tmp = copy.deepcopy(self.respy_base)
            respy_tmp.update_optim_paras(x_all_econ_eval_respy_old)
            fname = 'step.estimagic.ini'
            respy_tmp.write_out(fname)
            respy_ini_old_to_new(fname, True, fname)

    def set_derived_attributes(self):

        optim_paras = self.respy_base.get_attr('optim_paras')
        for k in ['start', 'step', 'current']:
            self.x_all_econ[k] = respy_spec_old_to_new(optim_paras)
            self.fval[k] = HUGE_FLOAT

        self.num_paras = len(self.x_all_econ['start'])

        self.logging_container = np.tile(np.nan, (self.num_paras + 1, 3))
        self.paras_free = ~np.array(optim_paras['paras_fixed'])

        self.x_free_econ_start = self.x_all_econ['start'][self.paras_free]

        if self.respy_base.get_attr('num_procs') > 1:
            worker = get_communicator(self.respy_base)
        else:
            worker = MISSING_INT

        self.mpi_setup = worker
