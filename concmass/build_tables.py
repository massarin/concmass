"""
Build concentration interpolators from Colossus models.

Usage:
    from concmass.build_tables import build_table
    from astropy.cosmology import Planck18

    table      = build_table()                                    # Planck18, diemer19 median
    table_mean = build_table(statistic='mean')
    table_d15  = build_table(model='diemer15')
    table_wmap = build_table(WMAP9, sigma8=0.817, ns=0.9608)

    # Restricted-range model (duffy08 valid for z <= 2):
    table_d08 = build_table(model='duffy08', mdef='200c', z_range=(0.0, 2.0))
"""
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Planck18 power-spectrum parameters (not stored in astropy cosmology objects)
_PLANCK18_SIGMA8 = 0.811
_PLANCK18_NS = 0.966


def _default_cosmology():
    from astropy.cosmology import Planck18
    return Planck18


def _set_colossus_cosmology(cosmo, sigma8, ns):
    from colossus.cosmology import cosmology as col_cosmo
    params = {
        "flat": float(cosmo.Ok0) == 0.0,
        "H0": float(cosmo.H0.value),
        "Om0": float(cosmo.Om0),
        "Ob0": float(cosmo.Ob0),
        "sigma8": sigma8,
        "ns": ns,
    }
    name = cosmo.name.lower()
    col_cosmo.addCosmology(name, **params)
    col_cosmo.setCosmology(name, persistence="r")


def build_table(
    cosmology=None,
    sigma8: float = None,
    ns: float = None,
    model: str = "diemer19",
    statistic: str = "median",
    mdef: str = "200c",
    M_range: tuple = (1e10, 1e16),
    z_range: tuple = (0.0, 5.0),
    n_M: int = 80,
    n_z: int = 40,
) -> RegularGridInterpolator:
    """
    Compute a concentration interpolator over an (n_M × n_z) (log10 M, z) grid.

    Parameters
    ----------
    cosmology   : astropy cosmology object; None → Planck18
    sigma8      : power spectrum normalisation; None → Planck18 (0.811)
    ns          : spectral index; None → Planck18 (0.966)
    model       : Colossus concentration model name (see concmass.MODELS)
    statistic   : 'median' or 'mean' (ignored by models that don't support it)
    mdef        : halo mass definition
    M_range     : (M_min, M_max) in Msun/h; clip to model validity range if needed
    z_range     : (z_min, z_max); clip to model validity range if needed
                  (e.g. z_range=(0, 2) for duffy08)
    n_M         : number of mass grid points (log-spaced)
    n_z         : number of redshift grid points (linear-spaced); must be >= 4 for cubic spline

    Returns
    -------
    RegularGridInterpolator  (log10_M, z) → log10_c, cubic, bounds_error=True
    """
    from colossus.halo import concentration as col_conc

    if cosmology is None:
        cosmology = _default_cosmology()
    if sigma8 is None:
        sigma8 = _PLANCK18_SIGMA8
    if ns is None:
        ns = _PLANCK18_NS

    _set_colossus_cosmology(cosmology, sigma8, ns)

    M_grid = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), n_M)
    z_grid = np.linspace(z_range[0], z_range[1], n_z)

    log10_c = np.empty((n_M, n_z))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, z in enumerate(z_grid):
            c, mask = col_conc.concentration(
                M_grid, mdef, z, model=model, statistic=statistic, range_return=True
            )
            if not mask.all():
                raise ValueError(
                    f"Colossus returned invalid concentrations for model={model!r}, "
                    f"mdef={mdef!r}, z={z:.3f}. Adjust M_range or z_range to the model's "
                    "validity range."
                )
            log10_c[:, j] = np.log10(c)

    return RegularGridInterpolator(
        (np.log10(M_grid), z_grid),
        log10_c,
        method="cubic",
        bounds_error=True,
    )
