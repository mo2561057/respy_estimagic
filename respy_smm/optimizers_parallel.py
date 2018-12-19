#!/usr/bin/env python
"""This module provides an interface for the parallel optimization. It allows to run the package
without the need to initialize MPIEXEC right from the beginning."""
import pickle as pkl
import os

from respy_smm.optimizers.optimizers_nag import run_nag


if __name__ == "__main__":

    infos = pkl.load(open('.infos.respy_smm.pkl', 'rb'))
    os.remove('.infos.respy_smm.pkl')

    args = list()
    for label in ['init_file', 'moments_obs', 'weighing_matrix', 'toolbox_spec']:
        args += infos[label]

    try:
        run_nag(*args)
    except StopIteration:
        pass
