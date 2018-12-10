"""This module hosts all functions required to interface with the NAG algorithms."""
import numpy as np
import pybobyqa
import dfols

from respy_smm.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.auxiliary_depreciation import process_shocks_bounds
from respy_smm import HUGE_INT
from functools import partial


def run_nag(fname, moments_obs, weighing_matrix, toolbox_spec):
    """This function serves as the interface to the NAG algorithms."""
    # Distribute user specification.
    algorithm = toolbox_spec['algorithm']
    max_evals = toolbox_spec['max_evals']

    # TODO: This is an auxiliary init file that follow the old RESPY specification of the
    #  initialization file. Otherwise, all kinds of errors are triggered.
    fname = '.smm.respy.ini'
    est_obj = SimulationBasedEstimationCls(fname, moments_obs, weighing_matrix, max_evals)
    x_all_econ_start = est_obj.get_attr('x_all_econ_start')

    # We need to construct the box based on the bounds specified in the initialization file.
    respy_base = est_obj.attr['respy_base']

    paras_fixed = np.array(respy_base.get_attr('optim_paras')['paras_fixed'])
    paras_bounds = respy_base.get_attr('optim_paras')['paras_bounds']

    # We need to ensure that the bounds are properly set for the parameters of the shock
    # distribution. If not specified, default bounds are set. We also need to set bounds for all
    # other parameters if these are not set in the initialization file.
    paras_bounds = process_shocks_bounds(paras_bounds)
    box = paras_bounds[~paras_fixed]

    # We lock in one evaluation at the starting values.
    x_free_econ_start = []
    for i, value in enumerate(x_all_econ_start):
        if not paras_fixed[i]:
            x_free_econ_start += [value]
    est_obj.criterion(x_free_econ_start)

    # TODO: This is a temporary bugfix for the fact that rhobeg is not set as outlined in the
    #   documentation. This is already noted in the CHANGELOG and will be available shortly.
    scaling_within_bounds, rhobeg = True, 0.1

    if algorithm == 'bobyqa':
        solve = pybobyqa.solve
    elif algorithm == 'dfols':
        solve = dfols.solve

    try:
        solve(partial(est_obj.criterion), np.array(x_free_econ_start),
              bounds=(box[:, 0], box[:, 1]), maxfun=HUGE_INT,
              scaling_within_bounds=scaling_within_bounds, rhobeg=rhobeg,
              objfun_has_noise=True)
    except StopIteration:
        pass
