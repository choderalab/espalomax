import pytest

def test_gat():
    import jax
    import espalomax as esp
    model = esp.nn.GraphAttentionNetwork(8, 3)
    graph = esp.Graph.from_smiles("C").homograph
    params = model.init(jax.random.PRNGKey(2666), graph)
    graph = model.apply(params, graph)
    assert graph.nodes.shape == (5, 8)

def test_janossy():
    import jax
    import jax.numpy as jnp
    import espalomax as esp
    graph = esp.Graph.from_smiles("C=O")
    heterograph = graph.heterograph
    h = jnp.zeros((5, 8))
    model = esp.nn.JanossyPooling(8, 3)
    params = model.init(jax.random.PRNGKey(2666), heterograph, h)

def test_parametrization():
    import jax
    import jax.numpy as jnp
    import espalomax as esp
    graph = esp.Graph.from_smiles("C=O")
    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3),
    )
    nn_params = model.init(jax.random.PRNGKey(2666), graph)
    ff_params = model.apply(nn_params, graph)
    print(ff_params)
