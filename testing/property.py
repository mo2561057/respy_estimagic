import random
import os

import numpy as np

from respy_smm.tests import test_integration
from respy_smm.tests import test_smm


while True:

    seed = random.randrange(1, 100000)
    print('\n', seed, '\n\n')

    np.random.seed(seed)

    for test in [test_integration.test_1, test_smm.test_1]:
        os.system('git clean -df')
        test()
