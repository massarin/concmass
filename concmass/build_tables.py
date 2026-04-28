"""
Build concentration interpolators from Colossus models.

Usage:
    from concmass.build_tables import build_table
    from astropy.cosmology import Planck18

    table        = build_table()                                   # Planck18 median defaults
    table_mean   = build_table(statistic='mean')
    table_wmap   = build_table(WMAP9, sigma8=0.817, ns=0.9608)
"""
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Planck18 power-spectrum parameters (not stored in astropy cosmology objects)
_PLANCK18_SIGMA8 = 0.811
_PLANCK18_NS = 0.966

_M_GRID = np.logspace(10, 16, 80)   # Msun/h
_Z_GRID = np.linspace(0, 5, 40)


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
) -> RegularGridInterpolator:
    """
    Compute a concentration interpolator over an 80×40 (log10 M, z) grid.

    Parameters
    ----------
    cosmology   : astropy cosmology object; None → Planck18
    sigma8      : power spectrum normalisation; None → Planck18 (0.811)
    ns          : spectral index; None → Planck18 (0.966)
    model       : Colossus concentration model name
    statistic   : 'median' or 'mean'
    mdef        : halo mass definition ('200c' is native for diemer19)

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

    log10_c = np.empty((len(_M_GRID), len(_Z_GRID)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, z in enumerate(_Z_GRID):
            c, mask = col_conc.concentration(
                _M_GRID, mdef, z, model=model, statistic=statistic, range_return=True
            )
            if not mask.all():
                raise ValueError(
                    f"Colossus returned invalid concentrations for model={model!r}, "
                    f"mdef={mdef!r}, z={z:.3f}. Check M/z grid against model validity range."
                )
            log10_c[:, j] = np.log10(c)

    return RegularGridInterpolator(
        (np.log10(_M_GRID), _Z_GRID),
        log10_c,
        method="cubic",
        bounds_error=True,
    )
