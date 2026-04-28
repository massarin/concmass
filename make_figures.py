"""
Generate diagnostic figures for concmass.
Run with: python make_figures.py
"""
import timeit
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from colossus.cosmology import cosmology as col_cosmo
from colossus.halo import concentration as col_conc

from concmass import conc
from concmass.build_tables import _PLANCK18_SIGMA8, _PLANCK18_NS, build_table

MDEF = "200c"
FIGURES = Path("figures")

col_cosmo.addCosmology(
    "planck18", flat=True,
    H0=float(Planck18.H0.value), Om0=float(Planck18.Om0), Ob0=float(Planck18.Ob0),
    sigma8=_PLANCK18_SIGMA8, ns=_PLANCK18_NS,
)
col_cosmo.setCosmology("planck18", persistence="r")

TABLE = build_table()


def colossus(M, z, statistic="median"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return col_conc.concentration(M, MDEF, z, model="diemer19", statistic=statistic)


# ---------------------------------------------------------------------------
# Figure 1 — speedup
# ---------------------------------------------------------------------------

def fig_speedup():
    sizes = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    t_col, t_interp = [], []

    for n in sizes:
        M = np.logspace(11, 15, n)
        z = np.linspace(0.1, 3.0, n)
        reps_col = max(1, min(5, 500 // n))
        reps_int = max(10, min(200, 5000 // n))
        tc = timeit.timeit(
            lambda: [colossus(np.array([mi]), float(zi)) for mi, zi in zip(M, z)],
            number=reps_col,
        ) / reps_col
        ti = timeit.timeit(lambda: conc("diemer19", M, z, TABLE), number=reps_int) / reps_int
        t_col.append(tc * 1e3)
        t_interp.append(ti * 1e3)
        print(f"n={n:6d}  colossus={tc*1e3:.2f}ms  interp={ti*1e3:.3f}ms  speedup={tc/ti:.0f}x")

    t_col, t_interp = np.array(t_col), np.array(t_interp)
    speedup = t_col / t_interp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.loglog(sizes, t_col, "o-", label="colossus (loop over z)")
    ax1.loglog(sizes, t_interp, "s-", label="concmass interpolator")
    ax1.set_xlabel("array size n")
    ax1.set_ylabel("wall time (ms)")
    ax1.legend()
    ax1.set_title("Timing")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogx(sizes, speedup, "^-", color="C2")
    ax2.axhline(100, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("array size n")
    ax2.set_ylabel("speedup (colossus / concmass)")
    ax2.set_title("Speedup factor")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = FIGURES / "speedup.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — residuals
# ---------------------------------------------------------------------------

def fig_residuals():
    M_grid = np.logspace(10.2, 15.8, 120)
    z_list = [0.0, 0.5, 1.0, 2.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(z_list)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for stat, ax in zip(["median", "mean"], axes):
        table = TABLE if stat == "median" else build_table(statistic="mean")
        for z, col in zip(z_list, colors):
            ref = colossus(M_grid, z, stat)
            got = conc("diemer19", M_grid, np.full_like(M_grid, z), table)
            rel_err = (got - ref) / ref
            ax.semilogx(M_grid, rel_err * 1e4, color=col, label=f"z={z}")

        ax.axhline(0, color="k", linewidth=0.6)
        ax.axhline(1, color="k", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.axhline(-1, color="k", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_xlabel(r"$M\ [M_\odot/h]$")
        ax.set_ylabel(r"$(c_\mathrm{interp} - c_\mathrm{colossus})\,/\,c_\mathrm{colossus}\ [\times 10^{-4}]$")
        ax.set_title(f"Residuals — {stat}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGURES / "residuals.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_speedup()
    fig_residuals()
