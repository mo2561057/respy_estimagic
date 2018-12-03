"""This module provides a unifying interface to the different algorithms of the package."""
from respy_smm.optimizers.optimizers_nag import run_nag


def optimize(init_file, moments_obs, weighing_matrix, toolbox, toolbox_spec):
    """This function routes the calls to the proper functions."""
    if toolbox not in ['nag']:
        raise NotImplementedError

    try:

        if toolbox in ['nag']:
            rslt = run_nag(init_file, moments_obs, weighing_matrix, toolbox_spec)

    except StopIteration:
        # TODO: This is not nicely done, it would be better to have a gentle shutdown.
        rslt = None


    return rslt
