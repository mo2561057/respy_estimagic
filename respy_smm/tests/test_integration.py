import pickle as pkl

import numpy as np

from respy.tests.codes.random_init import write_init_file
from respy.tests.codes.random_init import generate_init

from respy_smm.tests.auxiliary import get_ingredients
from respy_smm import optimize


def test_1():
    """This test ensures that an evaluation at the true values does result in a zero value of the
    criterion function."""
    constr = dict()
    constr['flag_interpolation'] = False
    constr['flag_myopic'] = False
    generate_init(constr)

    respy_obj, moments_obs, num_moments, weighing_matrix = get_ingredients('test.respy.ini')

    # TODO: This needs to be a general function once more optimization algorithms are available.
    toolbox = 'nag'
    toolbox_spec = dict()
    toolbox_spec['algorithm'] = 'bobyqa'
    toolbox_spec['max_evals'] = np.random.randint(1, 10)

    optimize('test.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)

    rslt = pkl.load(open('smm_monitoring.pkl', 'rb'))
    np.testing.assert_almost_equal(rslt['Step'].iloc[0], 0.0)


def test_2():
    """This test ensures that the version of the programm does not matter for the value of the
    criterion function."""
    constr = dict()
    constr['flag_interpolation'] = False
    constr['flag_myopic'] = False

    dict_ = generate_init(constr)

    # TODO: Would be nicer if we could just sample a random point ...
    respy_obj, moments_obs, num_moments, weighing_matrix = get_ingredients('test.respy.ini')

    num_agents_sim = np.random.randint(1, respy_obj.get_attr('num_agents_sim'))
    dict_['SIMULATION']['agents'] = num_agents_sim

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

        optimize('test.respy.ini', moments_obs, weighing_matrix, toolbox, toolbox_spec)

        rslt = pkl.load(open('smm_monitoring.pkl', 'rb'))
        if base is None:
            base = rslt['Step'].iloc[0]

        np.testing.assert_almost_equal(rslt['Step'].iloc[0], base)
