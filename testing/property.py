import random
import os

import pandas as pd
import numpy as np

from respy_smm.tests.test_integration import test_1
from respy_smm.tests.test_integration import test_2

counts = dict()
counts['is_success'] = 0
counts['num_total'] = 0

for _ in range(2):

    counts['num_total'] += 1

    seed = random.randrange(1, 100000)
    np.random.seed(seed)

    test_1()
    os.system('git clean -df')

    try:
        test_2()
        is_success = True
    except AssertionError:
        is_success = False

    if is_success:
        counts['is_success'] += 1

    df = pd.DataFrame.from_dict(counts, orient='index')
    df.to_string(open('testing.respy_smm.info', 'w'), index=True, justify='justify-all')

    os.system('git clean -df')
