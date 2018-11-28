"""This module contains functions to calculate the moments based on the observed and simulated
datasets."""
import pickle as pkl
import json

import pandas as pd


def get_moments(df, is_store=False):
    """This function computes the moments based on a dataframe."""
    moments = dict()
    moments['Choice Probability'] = dict()
    moments['Wage Distribution'] = dict()

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

    # We might want to store the data from the moments calculation for transfer to a different
    # estimation machine.
    if is_store:
        fname = 'moments.respy.'
        json.dump(moments, open(fname + 'json', 'w'), indent=4, sort_keys=True)
        pkl.dump(moments, open(fname + 'pkl', 'wb'))

    return moments
