"""This module hosts all functions required to interface with the NAG algorithms."""
import numpy as np
import pybobyqa
import dfols

from respy_smm.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.config_package import DEFAULT_BOUND
from respy_smm.config_package import HUGE_INT
from functools import partial


def run_nag(fname, moments_obs, weighing_matrix, toolbox_spec):
    """This function serves as the interface to the NAG algorithms."""
    # Distribute user specification.
    algorithm = toolbox_spec['algorithm']
    max_evals = toolbox_spec['max_evals']

    # TODO: This is an auxiliary init file that follow the old RESPY specification of the
    #  initialization file.
    fname = '.smm.respy.ini'
    est_obj = SimulationBasedEstimationCls(fname, moments_obs, weighing_matrix, max_evals)
    x_all_econ_start = est_obj.get_attr('x_all_econ_start')

    # We need to construct the box based on the bounds specified in the initialization file.
    respy_base = est_obj.attr['respy_base']

    paras_bounds = respy_base.get_attr('optim_paras')['paras_bounds']
    paras_fixed = respy_base.get_attr('optim_paras')['paras_fixed']

    # TODO: At this point we impose some default bounds for the parameters of the coeffs_shocks.
    #  This can be integrated in the package at a later stage. We impose that all standard
    #  deviations need to be larger than zero and that the coefficient of correlation is between
    #  zero and one.
    for i in range(43, 43 + 4):
        paras_bounds[i] = [1e-7, None]
    for i in range(43 + 4, 53):
        paras_bounds[i] = [-0.99, 0.99]

    # TODO: I need to be able to overwrite the bounds in the initialization file.
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
        solve(partial(est_obj.criterion), np.array(x_all_econ_start),
              bounds=(box[:, 0], box[:, 1]), maxfun=HUGE_INT,
              scaling_within_bounds=scaling_within_bounds, rhobeg=rhobeg,
              objfun_has_noise=True)
    except StopIteration:
        pass
