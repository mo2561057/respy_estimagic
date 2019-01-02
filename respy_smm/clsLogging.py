"""This module hosts the logging capabilities of the package."""
import pickle as pkl
import pandas as pd
import numpy as np

from respy.pre_processing.model_processing import convert_attr_dict_to_init_dict
from respy_smm.auxiliary_depreciation import respy_spec_old_to_new
from respy.pre_processing.model_processing import write_init_file
from respy_smm.auxiliary import format_column


class LoggingCls(object):

    def __init__(self):

        self.attr = dict()
        self.attr['weighing_matrix'] = None
        self.attr['paras_fixed'] = None
        self.attr['num_paras'] = None
        self.attr['df_info'] = None
        self.attr['num_steps'] = -1
        self.attr['num_evals'] = 0

        self.attr['info_dict'] = dict()
        self.attr['info_dict']['is_step'] = np.empty(0, dtype='bool')
        self.attr['info_dict']['fval'] = np.empty((0, 0))
        self.attr['info_dict']['paras'] = None

    def setup_information(self, weighing_matrix, max_evals, paras_fixed):
        """This method attaches some information that is constant for an estimation run but
        useful for further processing."""
        self.attr['paras_fixed'] = np.array(paras_fixed)
        self.attr['num_free'] = (~self.attr['paras_fixed']).sum()
        self.attr['weighing_matrix'] = weighing_matrix
        self.attr['num_paras'] = len(paras_fixed)
        self.attr['max_evals'] = max_evals

    @staticmethod
    def record_abort_eval(msg):
        """This method logs the early termination of an evaluation."""
        with open("smm_monitoring.log", 'a') as outfile:
            fmt_ = '    {:<25}\n\n'
            outfile.write(fmt_.format(*['WARNING']))
            fmt_ = '    - {:>25}\n'
            outfile.write(fmt_.format(msg))
            outfile.write('\n ' + '-' * 125 + '\n\n')

    def record(self, info_update, stats_obs, stats_sim, respy_smm, duration):
        """This method logs the progress of the estimation."""
        # Distribute class attributes and construct auxiliary objects.
        weighing_matrix = self.attr['weighing_matrix']
        paras_fixed = self.attr['paras_fixed']
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

            # TODO: This is required due to the different setup of the shock specification.
            #  Later, we can call respy_obj.write_out() again.
            init_dict = convert_attr_dict_to_init_dict(respy_smm.attr)
            x_all_econ_start = respy_spec_old_to_new(respy_smm.attr['optim_paras'])
            init_dict['SHOCKS']['coeffs'] = x_all_econ_start[43:55]
            write_init_file(init_dict, 'smm_monitoring.step.ini')
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

        with open(fname, 'a') as outfile:
            fmt_ = '    {:<25}\n\n'
            outfile.write(fmt_.format(*['OVERVIEW']))
            fmt_ = ' {:>25}{:>25}{:>25}{:>25}\n\n'
            outfile.write(fmt_.format(*['Evaluation', 'Step', 'Criterion', 'Seconds']))
            fmt_ = ' {:>25}{:>25}{:>25.5f}{:>25}\n'

            line = [self.attr["num_evals"], self.attr["num_steps"], self.attr["func"], duration]
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
                if not paras_fixed[i]:
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

            for i in range(len(stats_obs)):
                diff = np.abs(stats_obs[i] - stats_sim[i])
                line = [i, stats_obs[i], stats_sim[i], diff, weighing_matrix[i, i]]
                outfile.write(fmt_.format(*line))
            outfile.write('\n\n')

            outfile.write('\n ' + '-' * 125 + '\n\n')

        # TODO: Prototyping better diagnostics in an external ipython notebook. At the moment we
        #  store information on all evaluations but might need to reduce in the future as we
        #  write out a complete file each time.
        if True:
            if self.attr['info_dict']['paras'] is None:
                self.attr['info_dict']['paras'] = np.empty((self.attr['num_free'], 0))

            a, b = self.attr['info_dict']['is_step'], is_step
            self.attr['info_dict']['is_step'] = np.append(a, b)

            a, b = self.attr['info_dict']['fval'], self.attr['func']
            self.attr['info_dict']['fval'] = np.append(a, b)

            a = self.attr['info_dict']['paras']
            b = np.array(info_update[2:], ndmin=2).T[~self.attr['paras_fixed']]
            self.attr['info_dict']['paras'] = np.concatenate((a, b), axis=1)

            pkl.dump(self.attr['info_dict'], open('smm_monitoring.all.pkl', 'wb'))


logger_obj = LoggingCls()
