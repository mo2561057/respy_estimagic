"""This module contains some tests that make sure that the soon to be deprecated functions and
their workarounds are set up properly.
"""
import numpy as np

from respy_smm.auxiliary_depreciation import shocks_spec_new_to_old


def test_1():
    """This test just makes sure that the transformation of the shock matrix works properly."""
    for _ in range(10):
        x = np.random.uniform(size=10)
        arg_1, arg_2 = x[:4], np.array(shocks_spec_new_to_old(x))[[0, 4, 7, 9]]
        np.testing.assert_almost_equal(arg_1, arg_2)
