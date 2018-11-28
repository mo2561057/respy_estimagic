"""This module hosts all functions required to interface with the NAG algorithms."""
import numpy as np

import pybobyqa
import dfols

from ov_tools.smm_estimation.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from ov_tools.smm_estimation.auxiliary import get_starting_values_econ
from ov_tools.config_ov_tools import DEFAULT_BOUND
from ov_tools.config_ov_tools import HUGE_INT
from respy.clsRespy import PARAS_MAPPING
from functools import partial


def run_nag(fname, moments_obs, weighing_matrix, nag_spec):
    """This function serves as the interface to the NAG algorithms."""
    # Distribute user specification.
    optimizer = nag_spec['optimizer']
    max_evals = nag_spec['max_evals']

    est_obj = SimulationBasedEstimationCls(fname, moments_obs, weighing_matrix, max_evals=max_evals)
    x_free_econ_start = get_starting_values_econ(fname)

    # We need to construct the box based on the bounds specified in the initialization file.
    respy_base = est_obj.attr['respy_base']
    paras_bounds_reordered = respy_base.get_attr('optim_paras')['paras_bounds']
    paras_fixed = respy_base.get_attr('optim_paras')['paras_fixed']

    # TODO: This requirement is simply due to lack of a good parameter management in RESPY.
    paras_bounds = paras_bounds_reordered[:]
    for old, new in PARAS_MAPPING:
        paras_bounds[new] = paras_bounds_reordered[old]

    box = list()
    for i, bounds in enumerate(paras_bounds):
        if paras_fixed[i]:
            continue
        for j, bound in enumerate(bounds):
            if bound is None:
                bounds[j] = (- 1) ** (j + 1) * DEFAULT_BOUND

        box += [bounds]
    box = np.array(box)

    # We lock in one evaluation at the starting values.
    est_obj.criterion(True, x_free_econ_start)

    # TODO: This is a temporary bugfix for the fact that rhobeg is not set as outlined in the
    # documentation. This is already noted in the CHANGELOG and will be available shortly.
    scaling_within_bounds, rhobeg = True, 0.1

    if optimizer == 'bobyqa':
        solve = pybobyqa.solve
    elif optimizer == 'dfols':
        solve = dfols.solve

    try:
        solve(partial(est_obj.criterion, True), np.array(x_free_econ_start),
              bounds=(box[:, 0], box[:, 1]), maxfun=HUGE_INT,
              scaling_within_bounds=scaling_within_bounds, rhobeg=rhobeg,
              objfun_has_noise=True)
    except StopIteration:
        pass
