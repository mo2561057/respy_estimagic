import pickle as pkl

import numpy as np

from respy_smm.tests.auxiliary import run_regression_test
from respy_smm import PACKAGE_DIR


def test_1():
    tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.pkl', 'rb'))[:5]
    for i, test in enumerate(tests):
        seed, rslt = test
        np.testing.assert_equal(rslt, run_regression_test(seed))
