from concmass.models.diemer19 import conc_from_table

MODELS = [
    "bullock01",
    "duffy08",
    "klypin11",
    "prada12",
    "bhattacharya13",
    "dutton14",
    "diemer15_orig",
    "diemer15",
    "klypin16_m",
    "klypin16_nu",
    "ludlow16",
    "child18",
    "diemer19",
    "ishiyama21",
]

_DISPATCH = {m: conc_from_table for m in MODELS}


def conc(model, M, z, table, **kwargs):
    try:
        fn = _DISPATCH[model]
    except KeyError:
        raise ValueError(f"Unknown model {model!r}. Available: {sorted(_DISPATCH)}")
    return fn(M, z, table, **kwargs)
