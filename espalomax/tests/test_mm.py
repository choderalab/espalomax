def test_original_to_mixture_and_back():
    import espalomax as esp
    import numpy as onp
    phases = [0, 1]
    k = onp.random.uniform(0, 1, size=(10,))
    b = onp.random.uniform(0, 1, size=(10,))
    coefficients = esp.mm.original_to_linear_mixture(k, b, phases)
    k1, b1 = esp.mm.linear_mixture_to_original(coefficients, phases)
    assert onp.allclose(k, k1)
    assert onp.allclose(b, b1)

def test_mixture_to_original_and_back():
    import espalomax as esp
    import numpy as onp
    phases = [0, 1]
    coefficients = onp.random.randn(10, 2)
    k, b = esp.mm.linear_mixture_to_original(coefficients, phases)
    coefficients1 = esp.mm.original_to_linear_mixture(k, b, phases)
    assert onp.allclose(coefficients, coefficients1)

def test_snapshots_consistency():
    raise NotImplementedError
