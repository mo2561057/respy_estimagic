import numpy as np

import pytest

from respy_smm.auxiliary_depreciation import respy_spec_old_to_new
from respy.pre_processing.model_processing import write_init_file
from respy.tests.codes.random_init import generate_init
from respy_smm.tests.auxiliary import get_ingredients
from respy_smm import optimize
import respy


def test_1():
    """This test ensures that an evaluation at the true values does result in a zero value of the
    criterion function."""
    constr = dict()
    constr['flag_interpolation'] = False

    # This requires some additional work to align the disturbances across the MPI machines.
    constr['flag_parallelism_mpi'] = False

    # This ensures that we do not have the discount factor set to the boundary values which
    # results in an error in the setup of the NAG optimizers.
    constr['flag_myopic'] = False

    init_dict = generate_init(constr)

    _, moments_obs, _, weighing_matrix = get_ingredients('test.respy.ini')

    # TODO" This is a temporary fix where we move the old spec to the new one.
    optim_paras = respy.RespyCls('test.respy.ini').get_attr('optim_paras')
    init_dict['SHOCKS']['coeffs'] = respy_spec_old_to_new(optim_paras)[43:53]
    write_init_file(init_dict)

    # TODO: This needs to be a general function once more optimization algorithms are available.
    toolbox = 'nag'
    toolbox_spec = dict()
    toolbox_spec['algorithm'] = 'bobyqa'

    # In principle I would want multiple evaluations. However, starting the NAG optimizers
    # without bounds everywhere does result in extreme evaluations and the RESPY code is not
    # properly handling those. Maybe it should, but this is for another PR.
    toolbox_spec['max_evals'] = 1

    rslt = optimize('test.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)

    np.testing.assert_almost_equal(rslt['Step'].iloc[0], 0.0)


@pytest.mark.skip(reason="FORTRAN and PYTHON equality not maintained at this point")
def test_2():
    """This test ensures that the version of the program does not matter for the value of the
    criterion function."""
    constr = dict()
    constr['flag_interpolation'] = False

    dict_ = generate_init(constr)

    # TODO: Would be nicer if we could just sample a random point ...
    respy_obj, moments_obs, _, weighing_matrix = get_ingredients('test.respy.ini')

    num_agents_sim = np.random.randint(1, respy_obj.get_attr('num_agents_sim'))
    dict_['SIMULATION']['agents'] = num_agents_sim

    # TODO" This is a temporary fix where we move the old spec to the new one.
    optim_paras = respy.RespyCls('test.respy.ini').get_attr('optim_paras')
    dict_['SHOCKS']['coeffs'] = respy_spec_old_to_new(optim_paras)[43:53]
    write_init_file(dict_)

    base = None
    for version in ['PYTHON', 'FORTRAN']:

        dict_['PROGRAM']['version'] = version
        if version == 'PYTHON':
            dict_['PROGRAM']['threads'] = 1
            dict_['PROGRAM']['procs'] = 1
        else:
            dict_['PROGRAM']['procs'] = np.random.randint(1, 5)

        write_init_file(dict_)

        # TODO: This needs to be a general function once more optimization algorithms are available.
        toolbox = 'nag'
        toolbox_spec = dict()
        toolbox_spec['algorithm'] = 'bobyqa'
        toolbox_spec['max_evals'] = 1

        rslt = optimize('test.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)

        if base is None:
            base = rslt['Step'].iloc[0]

        np.testing.assert_almost_equal(rslt['Step'].iloc[0], base)
