"""This module hosts all functions required to interface with the SCIPY algorithms."""
from functools import partial

from scipy.optimize import minimize

from respy_smm.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.auxiliary import get_starting_values


def run_scipy(fname, moments_obs, weighing_matrix, method, max_eval):
    """This function serves as the interface to the SCIPY interfaces."""
    if method not in ['POWELL']:
        raise NotImplementedError

    # We need to determine if the optimizer does honor the bounds of the parameters directly or not.
    if method in ['POWELL']:
        is_econ = False
    else:
        raise NotImplementedError

    # We lock in an evaluation of the model at the starting values
    x_free_optim_start_scaled, scales = get_starting_values(fname)

    est_obj = SimulationBasedEstimationCls(fname, moments_obs, weighing_matrix, scales, max_eval)

    try:
        est_obj.criterion(is_econ, x_free_optim_start_scaled)
        criterion = partial(est_obj.criterion, is_econ)
        minimize(criterion, x_free_optim_start_scaled, method=method)
    except StopIteration:
        pass
