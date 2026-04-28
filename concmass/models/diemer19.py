import numpy as np
from scipy.interpolate import RegularGridInterpolator


def conc_diemer19(M, z, table: RegularGridInterpolator):
    """
    Parameters
    ----------
    M     : halo mass in Msun/h, scalar or array
    z     : redshift, scalar or array (broadcast with M)
    table : RegularGridInterpolator from build_table()
    """
    scalar = np.ndim(M) == 0 and np.ndim(z) == 0
    M = np.atleast_1d(np.asarray(M, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    M, z = np.broadcast_arrays(M, z)
    pts = np.stack([np.log10(M.ravel()), z.ravel()], axis=1)
    log10_c = table(pts).reshape(M.shape)
    c = 10.0 ** log10_c
    return float(c.squeeze()) if scalar else c
