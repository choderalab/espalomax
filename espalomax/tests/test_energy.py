import pytest

def test_get_energy():
    import jax
    import espalomax as esp
    graph = esp.Graph.from_smiles("C=C")

    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    geometry = esp.mm.get_geometry(graph.heterograph, x)

    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3),
    )
    nn_params = model.init(jax.random.PRNGKey(2666), graph)
    ff_params = model.apply(nn_params, graph)
    energy = esp.mm.get_energy(ff_params, geometry)

def test_get_d_energy_d_parameter():
    import jax
    import espalomax as esp
    graph = esp.Graph.from_smiles("C=C")

    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    geometry = esp.mm.get_geometry(graph.heterograph, x)

    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3),
    )
    nn_params = model.init(jax.random.PRNGKey(2666), graph)

    def get_energy(nn_params):
        ff_params = model.apply(nn_params, graph)
        energy = esp.mm.get_energy(ff_params, geometry)
        return energy

    d_energy_d_params = jax.grad(get_energy)(nn_params)
