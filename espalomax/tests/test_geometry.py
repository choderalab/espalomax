import pytest

def test_geometry():
    import jax
    import espalomax as esp
    graph = esp.Graph.from_smiles("C=C")
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    esp.mm.get_geometry(graph.heterograph, x)
