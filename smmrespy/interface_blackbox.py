"""This module contains all functions to interface the BLACKBOX algorithm."""
from functools import partial
import multiprocessing as mp
import atexit
import shutil
import os

try:
    from blackbox import search
except ModuleNotFoundError:
    pass

from ov_tools.smm_estimation.clsSimulationBasedEstimation import SimulationBasedEstimationCls
from ov_tools.smm_estimation.auxiliary import blackbox_wrapper
from ov_tools.smm_estimation.auxiliary import get_starting_values_econ
from ov_tools.smm_estimation.auxiliary import AggregatorCls
from ov_tools.smm_estimation.auxiliary import cleanup
from respy.clsRespy import PARAS_MAPPING


def run_blackbox(fname, moments_obs, weighing_matrix, blackbox_spec):
    """This function runs the BLACKBOX algorithm for global optimization."""
    # We need to ensure MPI4PY is available if requested.
    if blackbox_spec['strategy'] == 'mpi':
        try:
            import mpi4py  # noqa: F401
            assert 'PMI_SIZE' in os.environ.keys()
        except ModuleNotFoundError:
            raise NotImplementedError

    cleanup(True)

    # We will always store the best evaluation point in a new directory and first populate it
    # with an evaluation at the starting values.
    est_obj = SimulationBasedEstimationCls(fname, moments_obs, weighing_matrix)
    x_free_econ_start = get_starting_values_econ(fname)

    dirname = 'blackbox_best'
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

    os.chdir(dirname)
    est_obj.criterion(True, x_free_econ_start)
    os.chdir('../')

    # Start aggregation process that collects evaluations throughout.
    is_finished = mp.Event()
    aggregator_process = mp.Process(target=AggregatorCls().run, daemon=True, args=(is_finished, ))
    aggregator_process.start()
    atexit.register(aggregator_process.terminate)

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
        box.append(bounds)

    if 'restart' in blackbox_spec.keys():
        is_restart = blackbox_spec['restart']
    else:
        is_restart = False

    strategy = blackbox_spec['strategy']
    batch = blackbox_spec['batch']
    m = blackbox_spec['m']
    n = blackbox_spec['n']

    points = search(partial(blackbox_wrapper, est_obj), box, n, m, batch, strategy,
                    is_restart=is_restart)

    # We now signal to the aggregator that it is time to start with the last iteration and wait
    # for it to finish.
    is_finished.set()
    aggregator_process.join()

    cleanup()

    return points
