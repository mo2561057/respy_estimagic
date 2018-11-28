import random
import os

import numpy as np

from respy_smm.tests.test_integration import test_1
from respy_smm.tests.test_integration import test_2

for _ in range(1):

    seed = random.randrange(1, 100000)
    np.random.seed(seed)

    test_1()
    os.system('git clean -df')

    test_2()
    os.system('git clean -df')
