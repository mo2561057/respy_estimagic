import numpy as np
from respy_smm.MaximumLikelihoodEstimation import MaximumLikelihoodEstimationCls
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy_smm.auxiliary_depreciation import respy_obj_from_new_init
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.tests.codes.auxiliary import write_lagged_start
from respy.tests.codes.auxiliary import write_edu_start
from respy.tests.codes.auxiliary import write_draws
from respy.tests.codes.auxiliary import write_types
import f2py_interface as respy_f2py
from respy.pre_processing.model_processing import write_init_file
from respy_smm.auxiliary_depreciation import respy_ini_old_to_new
from respy_smm.auxiliary_depreciation import x_all_econ_new_to_old
from respy_smm.tests.auxiliary import get_observed_sample, get_random_init
from respy_smm.tests.auxiliary import get_random_point

from respy_smm.tests.auxiliary import get_observed_sample, get_random_point, get_random_init, \
    mock_get_weighing_matrix, mock_get_moments
from respy_smm.SimulationBasedEstimation import SimulationBasedEstimationCls
from respy_smm.auxiliary import get_mpi


init_dict_old = get_random_init()
df = get_observed_sample()

rslt = list()
for num_procs in [1, 3]:
    init_dict_old['PROGRAM']['procs'] = num_procs

    write_init_file(init_dict_old)
    respy_ini_old_to_new('test.respy.ini', True, 'test.respy.ini')

    est_obj = MaximumLikelihoodEstimationCls(*('test.respy.ini', 3))
    est_obj.terminate(is_gentle=True)

    rslt.append(est_obj.fval['start'])


# TODO: This is not running properly in parallelism as I need to extract the remaining randomness.
np.testing.assert_almost_equal(*rslt)

df = get_observed_sample()

weighing_matrix = mock_get_weighing_matrix(df)
moments_obs = mock_get_moments(df)

for num_procs in [1, 3]:
    init_dict_old['PROGRAM']['procs'] = num_procs

    write_init_file(init_dict_old)
    respy_ini_old_to_new('test.respy.ini', True, 'test.respy.ini')

    args = ('test.respy.ini', moments_obs, weighing_matrix, mock_get_moments, 5)
    est_obj = SimulationBasedEstimationCls(*args)
    est_obj.evaluate(get_random_point())
    est_obj.terminate(is_gentle=True)
