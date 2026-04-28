from concmass.models.diemer19 import conc_diemer19

_DISPATCH = {
    "diemer19": conc_diemer19,
}


def conc(model, M, z, table, **kwargs):
    try:
        fn = _DISPATCH[model]
    except KeyError:
        raise ValueError(f"Unknown model {model!r}. Available: {sorted(_DISPATCH)}")
    return fn(M, z, table, **kwargs)
