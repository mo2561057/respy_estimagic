"""This module provides the fixtures for the PYTEST runs."""
import tempfile
import os

import numpy as np
import pytest


@pytest.fixture(scope='function', autouse=True)
def set_seed():
    """Each test is executed with the same random seed."""
    np.random.seed(1423)


@pytest.fixture(scope='function', autouse=True)
def fresh_directory():
    """Each test is executed in a fresh directory."""
    os.chdir(tempfile.mkdtemp())


@pytest.fixture(scope='function', autouse=True)
def set_environment():
    """The testing setup can be determine."""
    os.environ['TESTING'] = 'True'
