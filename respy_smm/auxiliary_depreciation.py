"""This module contains functions that will be depreciated once the scheduled improvements in
RESPY are completed."""
import numpy as np

from respy.python.shared.shared_auxiliary import get_optim_paras
from respy_smm.config_package import DEFAULT_BOUND


def process_shocks_bounds(paras_bounds):

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

    sds = shock_spec_new[:4]
    rho = shock_spec_new[4:]

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


# TODO: Can I better align the interfaces to the two functions, this is now a all out preacemen t?
def respy_spec_old_to_new(optim_paras, num_paras):

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
