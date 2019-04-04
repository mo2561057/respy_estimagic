"""This module contains all functions that allow to construct the weighing matrix."""
from collections import OrderedDict
import numpy as np
import pickle as pkl
import json

import pandas as pd


def moments_dict_to_list(moments_dict):
    """This function constructs a list of available moments based on the moment dictionary."""
    moments_list = []
    for group in moments_dict.keys():
        for period in moments_dict[group].keys():
            moments_list.extend(moments_dict[group][period])
    return moments_list


def get_weighing_matrix(df_base, num_boots, num_agents_smm, is_store=False):
    """This function constructs the weighing matrix."""
    # Ensure reproducibility
    np.random.seed(123)

    # Distribute clear baseline information.
    index_base = df_base.index.get_level_values('Identifier').unique()
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

    # We need to deal with the case that the variance for some moments is zero. This can happen
    # for example for the choice probabilities if early in the lifecycle nobody is working. This
    # will happen with the current setup of the final schooling moments, where we simply create a
    # grid of potential final schooling levels and fill it with zeros if not observed in the
    # data. We just replace it with the weight of an average moment.
    is_zero = moments_var <= 1e-10

    # TODO: As this only applies to the moments that are bounded between zero and one,
    #  this number comes up if I calculate the variance of random uniform variables.
    moments_var[is_zero] = 0.1

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


def get_moments(df, is_store=False):
    """This function computes the moments based on a dataframe."""
    moments = OrderedDict()
    for group in ['Wage Distribution', 'Choice Probability', 'Final Schooling']:
        moments[group] = OrderedDict()

    # We now add descriptive statistics of the wage distribution. Note that there might be
    # periods where there is no information available. In this case, it is simply not added to
    # the dictionary.
    info = df['Wage'].groupby('Period').describe().to_dict()
    for period in sorted(df['Period'].unique().tolist()):
        if pd.isnull(info['std'][period]):
            continue
        moments['Wage Distribution'][period] = []
        for label in ['mean', 'std']:
            moments['Wage Distribution'][period].append(info[label][period])

    # We first compute the information about choice probabilities. We need to address the case
    # that a particular choice is not taken at all in a period and then these are not included in
    # the dictionary. This cannot be addressed by using categorical variables as the categories
    # without a value are not included after the groupby operation.
    info = df['Choice'].groupby('Period').value_counts(normalize=True).to_dict()
    for period in sorted(df['Period'].unique().tolist()):
        moments['Choice Probability'][period] = []
        for choice in range(1, 5):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments['Choice Probability'][period].append(stat)

    # We add the relative share of the final levels of schooling. Note that we simply loop over a
    # large range of maximal schooling levels to avoid the need to pass any further details from
    # the package to the function such as the initial and maximal level of education at this point.
    info = df['Years_Schooling'].groupby('Identifier').max().value_counts(normalize=True).to_dict()
    for edu_max in range(30):
        try:
            stat = info[edu_max]
        except KeyError:
            stat = 0
        moments['Final Schooling'][edu_max] = [stat]

    # We might want to store the data from the moments calculation for transfer to a different
    # estimation machine.
    if is_store:
        fname = 'moments.respy.'
        json.dump(moments, open(fname + 'json', 'w'), indent=4, sort_keys=True)
        pkl.dump(moments, open(fname + 'pkl', 'wb'))

    return moments
