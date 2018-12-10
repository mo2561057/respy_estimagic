"""This module hosts the logging capabilities of the package."""
import pandas as pd
import numpy as np

from respy_smm.auxiliary import format_column


class LoggingCls(object):

    def __init__(self):

        self.attr = dict()
        self.attr['paras_fixed'] = None
        self.attr['num_paras'] = None
        self.attr['df_info'] = None
        self.attr['num_evals'] = 0
        self.attr['num_steps'] = 0

    def setup_information(self, num_paras, max_evals, paras_fixed):

        logger_obj.attr['paras_fixed'] = paras_fixed
        self.attr['num_paras'] = num_paras
        self.attr['max_evals'] = max_evals

    @staticmethod
    def record_abort_eval(msg):
        """This method logs the early termination of """
        with open("smm_monitoring.log", 'a') as outfile:
            fmt_ = '    {:<25}\n\n'
            outfile.write(fmt_.format(*['WARNING']))
            fmt_ = '    - {:>25}\n'
            outfile.write(fmt_.format(msg))
            outfile.write('\n ' + '-' * 125 + '\n\n')

    def record(self, info_update, stats_obs, stats_sim, weighing_matrix, respy_smm, duration):
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

        with open(fname, 'a') as outfile:
            fmt_ = '    {:<25}\n\n'
            outfile.write(fmt_.format(*['OVERVIEW']))
            fmt_ = ' {:>25}{:>25}{:>25}{:>25}\n\n'
            outfile.write(fmt_.format(*['Evaluation', 'Step', 'Criterion', 'Seconds']))
            fmt_ = ' {:>25}{:>25}{:>25.5f}{:>25.5f}\n'

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

            for i in range(len(stats_obs)):
                diff = np.abs(stats_obs[i] - stats_sim[i])
                line = [i, stats_obs[i], stats_sim[i], diff, weighing_matrix[i, i]]
                outfile.write(fmt_.format(*line))
            outfile.write('\n\n')

            outfile.write('\n ' + '-' * 125 + '\n\n')


logger_obj = LoggingCls()
