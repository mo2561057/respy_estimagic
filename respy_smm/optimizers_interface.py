"""This module provides a unifying interface to the different algorithms of the package."""
import pickle as pkl
import subprocess
import respy
import sys

from respy_smm.optimizers.optimizers_nag import run_nag
from respy_smm import PACKAGE_DIR


def optimize(init_file, moments_obs, weighing_matrix, toolbox, toolbox_spec):
    """This function routes the calls to the proper functions."""

    if toolbox not in ['nag']:
        raise NotImplementedError


    respy_obj = respy.RespyCls(init_file)

    num_procs = respy_obj.get_attr('num_procs')

    try:
        if num_procs == 1:
            run_nag(init_file, moments_obs, weighing_matrix, toolbox_spec)


        else:

            infos = dict()
            infos['init_file'] = init_file
            infos['moments_obs'] = moments_obs
            infos['weighing_matrix'] = weighing_matrix
            infos['toolbox_spec'] = toolbox_spec

            pkl.dump(infos, open('.infos.respy_smm.pkl', 'wb'))

            cmd = ['mpiexec', '-n', '1', sys.executable, PACKAGE_DIR + '/optimizers_parallel.py']
            subprocess.check_call(cmd)

    except StopIteration:
        pass

    rslt = pkl.load(open('smm_monitoring.pkl', 'rb'))

    return rslt