import pickle as pkl

import numpy as np
import pybobyqa

from respy_smm.auxiliary_depreciation import respy_obj_from_new_init, process_shocks_bounds


def wrapper_pybobyqa(crit_func, start_values, **kwargs):

    try:
        pybobyqa.solve(crit_func, start_values, **kwargs)
    except StopIteration:
        pass

    return pkl.load(open('monitoring.estimagic.pkl', 'rb'))


def get_box_bounds(init_file):

    respy_obj = respy_obj_from_new_init(init_file)
    paras_free = ~np.array(respy_obj.get_attr('optim_paras')['paras_fixed'])
    paras_bounds = respy_obj.get_attr('optim_paras')['paras_bounds']

    # We need to ensure that the bounds are properly set for the parameters of the shock
    # distribution. If not specified, default bounds are set. We also need to set bounds for all
    # other parameters if these are not set in the initialization file.
    paras_bounds = process_shocks_bounds(paras_bounds)
    box = paras_bounds[paras_free]

    return box