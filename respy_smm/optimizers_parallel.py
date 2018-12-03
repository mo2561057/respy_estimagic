#!/usr/bin/env python
import pickle as pkl
import os

from respy_smm.optimizers.optimizers_nag import run_nag



if __name__ == "__main__":
    infos = pkl.load(open('.infos.respy_smm.pkl', 'rb'))

    os.remove('.infos.respy_smm.pkl')
    init_file = infos['init_file']
    moments_obs = infos['moments_obs']
    weighing_matrix = infos['weighing_matrix']
    toolbox_spec = infos['toolbox_spec']

    try:
        run_nag(init_file, moments_obs, weighing_matrix, toolbox_spec)
    except StopIteration:
        pass
