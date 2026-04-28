# concmass

Fast halo concentration interpolator built from [Colossus](https://bdiemer.bitbucket.io/colossus/) models. Cubic interpolation on a pre-built in-memory grid gives ~1e-4 relative precision vs. direct Colossus calls, at 100–700× lower cost.

## Install

```
pip install -e .
```

## Usage

```python
from concmass import conc, conc_diemer19
from concmass.build_tables import build_table
import numpy as np

# build an interpolator (Planck18 + median by default)
table = build_table()
table_mean = build_table(statistic="mean")

# scalar or array
c = conc("diemer19", 1e13, 0.5, table)
c = conc("diemer19", np.logspace(11, 15, 100), 1.0, table_mean)

# or call the model function directly
c = conc_diemer19(M, z, table)
```

## Building tables

Pass any astropy cosmology object. `sigma8` and `ns` must be provided explicitly — astropy cosmology objects do not carry power-spectrum parameters.

```python
from astropy.cosmology import WMAP9
from concmass.build_tables import build_table

table = build_table(WMAP9, sigma8=0.817, ns=0.9608)
table = build_table(WMAP9, sigma8=0.817, ns=0.9608, statistic="mean")
```

## Performance

Cubic spline interpolation on an 80 × 40 (log M, z) grid. Colossus only accepts scalar z so it must loop over redshifts; the interpolator vectorises natively over arbitrary (M, z) arrays.

![speedup](figures/speedup.png)

## Precision

Max relative error vs. direct Colossus calls is ~1e-4 across the full grid (M ∈ [1e10, 1e16] M☉/h, z ∈ [0, 5]).

![residuals](figures/residuals.png)

## Tests

```
pytest tests/
```

Equivalency tests check `rtol=1e-4` against Colossus for both `median` and `mean` statistics across 12 mass values × 5 redshifts. A speedup test asserts >100× at n=1000.
