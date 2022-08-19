import pytest

def test_constant_padding():
    import espalomax as esp
    import numpy as onp
    data = []
    for idx in range(1, 6):
        g = esp.Graph.from_smiles("C=C"*idx)
        x = onp.zeros((1, g.n_atoms, 3))
        u = onp.zeros((1, ))
        data.append((g, x, u))
    dataloader = esp.data.PadToConstantDataLoader(data, 2)
    g, x, u, m = next(iter(dataloader))
