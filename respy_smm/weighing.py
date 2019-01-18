"""This module contains all functions that allow to construct the weighing matrix."""
import numpy.ma as ma
import pickle as pkl
import numpy as np

from respy_smm.auxiliary import moments_dict_to_list
from respy_smm.moments import get_moments


def get_weighing_matrix(df_base, num_boots, num_agents_smm, is_store=False):
    """This function constructs the weighing matrix."""
    # Ensure reproducibility
    np.random.seed(123)

    # Distribute clear baseline information.
    index_base = df_base.index.get_level_values('Identifier').unique()
    num_periods = df_base['Period'].nunique()
    moments_base = get_moments(df_base)
    num_boots_max = num_boots * 2

    # Initialize counters to keep track of attempts.
    num_valid, num_attempts, moments_sample = 0, 0, []

    while True:

        try:
            sample_ids = np.random.choice(index_base, num_agents_smm, replace=False)
            moments_boot = get_moments(df_base.loc[(sample_ids, slice(None)), :])

            # We want to confirm that we have valid values for all required moments that we were
            # able to calculate on the observed dataset.
            for group in ['Choice Probability', 'Wage Distribution']:
                for period in moments_base[group].keys():
                    if period not in moments_boot[group].keys():
                        raise NotImplementedError

            moments_sample.append(moments_boot)

            num_valid += 1
            if num_valid == num_boots:
                break

            # We need to terminate attempts that just seem to not work.
            if num_attempts > num_boots_max:
                raise NotImplementedError("... too many samples needed for matrix")

        except NotImplementedError:
            continue

        num_attempts += 1

    # Construct the weighing matrix based on the sampled moments.
    stats = []
    for moments_boot in moments_sample:
        stats.append(moments_dict_to_list(moments_boot))

    moments_var = np.array(stats).T.var(axis=1)

    # We need to deal with the case that the standard deviation for the choice probabilities. This
    # can happen for some of the choice probabilities. In particular early in the life-cycle
    # there is nobody working for example. At the moment, we simply replace them with the weight
    # of an average moment.
    moments_var_prob = moments_var[:num_periods * 4]
    is_zero = moments_var_prob <= 1e-10
    moments_var_prob[is_zero] = np.mean(ma.masked_array(moments_var_prob, mask=is_zero))

    if np.all(is_zero):
        raise NotImplementedError('... all variances are zero')
    if np.any(is_zero):
        print('... some variances are zero')

    weighing_matrix = np.diag(moments_var ** (-1))

    # Store information for further processing and inspection.
    if is_store:
        fname = 'weighing.respy.'
        np.savetxt(fname + 'txt', weighing_matrix, fmt='%35.15f')
        pkl.dump(weighing_matrix, open(fname + 'pkl', 'wb'))

    return weighing_matrix
