"""
Equivalency tests: concmass interpolator vs. direct Colossus calls.
Cubic interpolation on an 80×40 grid achieves max relative error ~1e-4.
"""
import timeit
import warnings

import numpy as np
import pytest
from astropy.cosmology import Planck18
from colossus.cosmology import cosmology as col_cosmo
from colossus.halo import concentration as col_conc

from concmass import conc
from concmass.build_tables import _PLANCK18_NS, _PLANCK18_SIGMA8, build_table

RTOL = 1e-4
MDEF = "200c"


@pytest.fixture(scope="module")
def tables():
    return {
        "median": build_table(),
        "mean":   build_table(statistic="mean"),
    }


@pytest.fixture(scope="module", autouse=True)
def set_colossus_cosmology():
    col_cosmo.addCosmology(
        "planck18", flat=True,
        H0=float(Planck18.H0.value), Om0=float(Planck18.Om0), Ob0=float(Planck18.Ob0),
        sigma8=_PLANCK18_SIGMA8, ns=_PLANCK18_NS,
    )
    col_cosmo.setCosmology("planck18", persistence="r")


def _colossus(M, z, statistic):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return col_conc.concentration(M, MDEF, z, model="diemer19", statistic=statistic)


M_CASES = np.logspace(10.5, 15.5, 12)
Z_CASES = [0.0, 0.5, 1.0, 2.0, 4.0]


@pytest.mark.parametrize("z", Z_CASES)
def test_median_vs_colossus(tables, z):
    got = conc("diemer19", M_CASES, np.full_like(M_CASES, z), tables["median"])
    ref = _colossus(M_CASES, z, "median")
    np.testing.assert_allclose(got, ref, rtol=RTOL)


@pytest.mark.parametrize("z", Z_CASES)
def test_mean_vs_colossus(tables, z):
    got = conc("diemer19", M_CASES, np.full_like(M_CASES, z), tables["mean"])
    ref = _colossus(M_CASES, z, "mean")
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_scalar_returns_float(tables):
    assert isinstance(conc("diemer19", 1e13, 0.5, tables["median"]), float)


def test_unknown_model_raises(tables):
    with pytest.raises(ValueError, match="Unknown model"):
        conc("bogus", 1e12, 0.0, tables["median"])


def test_out_of_bounds_raises(tables):
    with pytest.raises(Exception):
        conc("diemer19", 1e9, 0.0, tables["median"])


def test_speedup_vs_colossus(tables):
    n = 1000
    M = np.logspace(11, 15, n)
    z = np.linspace(0.1, 3.0, n)
    table = tables["median"]

    t_col = timeit.timeit(
        lambda: [_colossus(np.array([mi]), float(zi), "median") for mi, zi in zip(M, z)],
        number=3,
    ) / 3
    t_interp = timeit.timeit(lambda: conc("diemer19", M, z, table), number=30) / 30

    speedup = t_col / t_interp
    assert speedup > 100, f"Expected >100x speedup, got {speedup:.1f}x"
