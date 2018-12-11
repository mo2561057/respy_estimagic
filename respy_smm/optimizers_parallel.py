#!/usr/bin/env python
"""This module provides an interface for the parallel optimization. It allows to run the package
without the need to initialize MPIEXEC right from the beginning."""
import pickle as pkl
import os

from respy_smm.optimizers.optimizers_nag import run_nag


if __name__ == "__main__":

    infos = pkl.load(open('.infos.respy_smm.pkl', 'rb'))
    os.remove('.infos.respy_smm.pkl')

    weighing_matrix = infos['weighing_matrix']
    toolbox_spec = infos['toolbox_spec']
    moments_obs = infos['moments_obs']
    init_file = infos['init_file']

    try:
        run_nag(init_file, moments_obs, weighing_matrix, toolbox_spec)
    except StopIteration:
        pass
