"""This module hosts all functions required to interface with the NAG algorithms."""
import numpy as np
import pybobyqa
import dfols

from respy_smm.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.auxiliary_depreciation import process_shocks_bounds
from respy_smm import HUGE_INT


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
    respy_base = est_obj.get_attr('respy_base')
    paras_free = ~np.array(respy_base.get_attr('optim_paras')['paras_fixed'])
    paras_bounds = respy_base.get_attr('optim_paras')['paras_bounds']

    # We need to ensure that the bounds are properly set for the parameters of the shock
    # distribution. If not specified, default bounds are set. We also need to set bounds for all
    # other parameters if these are not set in the initialization file.
    paras_bounds = process_shocks_bounds(paras_bounds)
    x_free_econ_start = x_all_econ_start[paras_free]
    box = paras_bounds[paras_free]

    # We lock in one evaluation at the starting values.
    est_obj.criterion(x_free_econ_start)

    if algorithm == 'bobyqa':
        solve = pybobyqa.solve
    elif algorithm == 'dfols':
        solve = dfols.solve

    # The NAG algorithms do not appear to check whether the staring values are actually inside the
    # bounds.
    for i in range(len(x_free_econ_start)):
        start, bound = x_free_econ_start[i], box[i, :]
        if not bound[0] < start < bound[1]:
            msg = '... starting value for parameter {:} out of bounds'.format(i)
            raise AssertionError(msg)

    # TODO: This is a temporary bugfix for the fact that rhobeg is not set as outlined in the
    #   documentation. This is already noted in the CHANGELOG and will be available shortly.
    scaling_within_bounds, rhobeg = True, 0.1

    # TODO: This is a first attempt to, at least over time, uncouple the interfaces. Meaning that
    #  users can just rely on the documentation to the optimizers and head down the specifications.
    kwargs = dict()
    kwargs['scaling_within_bounds'] = scaling_within_bounds
    kwargs['bounds'] = (box[:, 0], box[:, 1])
    kwargs['objfun_has_noise'] = True
    kwargs['maxfun'] = HUGE_INT
    kwargs['rhobeg'] = rhobeg

    try:
        solve(est_obj.criterion, x_free_econ_start, **kwargs)
    except StopIteration:
        pass
