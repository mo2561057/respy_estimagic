"""This module contains functions that will be depreciated once the scheduled improvements in
RESPY are completed."""
import os

import numpy as np

from respy.python.shared.shared_auxiliary import coeffs_to_cholesky
from respy.pre_processing.model_processing import write_init_file
from respy.pre_processing.model_processing import read_init_file
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.clsRespy import RespyCls

from respy_smm import DEFAULT_BOUND


def process_shocks_bounds(paras_bounds):
    """This function ensures that proper bounds are set of the standard deviations and the
    coefficients of correlation."""
    num_paras = len(paras_bounds)

    paras_bounds_new = np.tile(np.nan, (num_paras, 2))
    for i in range(num_paras):
        paras_bounds_new[i, :] = paras_bounds[i]
    paras_bounds = paras_bounds_new

    # We need to ensure that standard deviations are positive.
    stds = paras_bounds[43:47, :]
    stds[np.isnan(stds[:, 0]), 0] = 0.01

    # We need to ensure that the coefficients fo correlation are bound between -1 and 1.
    rhos = paras_bounds[47:53, :]
    for i in range(2):
        bound = rhos[:, i]
        bound[np.isnan(bound)] = (-1.0) ** i * -0.99

    # If not set already, all parameters need bounds.
    paras_bounds[np.isnan(paras_bounds[:, 0]), 0] = -DEFAULT_BOUND
    paras_bounds[np.isnan(paras_bounds[:, 1]), 1] = +DEFAULT_BOUND

    return paras_bounds


def shocks_spec_new_to_old(shock_spec_new):
    """This function transfers the specification of the shocks from the new setup to the original
    one."""
    sds, rho = shock_spec_new[:4], shock_spec_new[4:]

    shocks_cov = np.zeros((4, 4))

    shocks_cov[1, 0] = rho[0] * sds[1] * sds[0]
    shocks_cov[2, 0] = rho[1] * sds[2] * sds[0]
    shocks_cov[2, 1] = rho[2] * sds[2] * sds[1]
    shocks_cov[3, 0] = rho[3] * sds[3] * sds[0]
    shocks_cov[3, 1] = rho[4] * sds[3] * sds[1]
    shocks_cov[3, 2] = rho[5] * sds[3] * sds[2]

    np.fill_diagonal(shocks_cov, sds ** 2)

    shocks_cov = shocks_cov + shocks_cov.T - np.diag(shocks_cov.diagonal())

    np.testing.assert_almost_equal(shocks_cov, shocks_cov.T)

    shocks_cov[np.diag_indices(4)] **= 0.5

    # This is called shock_coeffs in the original RESPY codes.
    shock_spec_old = shocks_cov[np.triu_indices(len(shocks_cov))].tolist()

    return shock_spec_old


def respy_spec_old_to_new(optim_paras):
    """This function transfers the specification of the shocks from the old setup to the new one."""
    num_paras = len(optim_paras['paras_fixed'])
    x_all_econ_start = get_optim_paras(optim_paras, num_paras, 'all', True)
    shocks_cov = optim_paras['shocks_cholesky'].dot(optim_paras['shocks_cholesky'].T)

    sds = np.sqrt(np.diag(shocks_cov))
    rho = np.tile(np.nan, 6)
    rho[0] = shocks_cov[1, 0] / (sds[1] * sds[0])
    rho[1] = shocks_cov[2, 0] / (sds[2] * sds[0])
    rho[2] = shocks_cov[2, 1] / (sds[2] * sds[1])
    rho[3] = shocks_cov[3, 0] / (sds[3] * sds[0])
    rho[4] = shocks_cov[3, 1] / (sds[3] * sds[1])
    rho[5] = shocks_cov[3, 2] / (sds[3] * sds[2])

    x_all_econ_start[43:53] = np.concatenate((sds, rho))

    return x_all_econ_start


def respy_obj_from_new_init(init_file, is_keep=False, file_name='.smm.respy.ini'):
    """This function creates a new initialization file that allows to read it directly with the
    core RESPY package"""
    init_dict = read_init_file(init_file)

    shock_spec_new = init_dict['SHOCKS']['coeffs']
    shock_spec_old = shocks_spec_new_to_old(shock_spec_new)
    init_dict['SHOCKS']['coeffs'] = shock_spec_old

    try:
        coeffs_to_cholesky(shock_spec_old)
    except np.linalg.linalg.LinAlgError:
        raise SystemExit(' ... correlation matrix not positive semidefinite')

    write_init_file(init_dict, file_name=file_name)
    respy_obj = RespyCls(file_name)

    if not is_keep:
        os.unlink(file_name)

    return respy_obj


def respy_ini_old_to_new(init_file, is_keep=False, file_name='.smm.respy.ini'):
    """This function creates a new initialization file that allows to read it directly with the
    core RESPY package"""
    init_dict = read_init_file(init_file)

    respy_obj = RespyCls(init_file)

    # TODO: This is another case where the interface is flawed
    optim_paras = respy_obj.get_attr('optim_paras')
    init_dict['SHOCKS']['coeffs'] = respy_spec_old_to_new(optim_paras)[43:53]
    write_init_file(init_dict, file_name)
    if not is_keep:
        os.unlink(file_name)

    return respy_obj


def x_all_econ_new_to_old(x_all_econ_new):
    x_all_econ_old = x_all_econ_new.copy()
    x_all_econ_old[43:53] = shocks_spec_new_to_old(x_all_econ_new[43:53])

    # We need to pass in the Cholesky factor.
    shocks_cholesky = coeffs_to_cholesky(x_all_econ_old[43:53])
    x_all_econ_old[43:53] = shocks_cholesky[np.tril_indices(4)]

    return x_all_econ_old
