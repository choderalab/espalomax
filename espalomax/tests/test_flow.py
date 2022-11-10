def test_janossy_parameter_convert():
    import espalomax as esp
    polynomial_parameters = esp.flow.get_polynomial_parameters()


def test_poly_eval():
    import jax
    import espalomax as esp
    polynomial_parameters = esp.flow.get_polynomial_parameters()

    from functools import partial
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("CC")

    graph = esp.Graph.from_openff_molecule(molecule)
    model = esp.nn.Parametrization(
        representation=esp.nn.GraphSageModel(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3, out_features=polynomial_parameters),

    )
    nn_params = model.init(jax.random.PRNGKey(2666), graph)
    flow_params = model.apply(nn_params, graph)
    ff_params = esp.flow.eval_polynomial(0.5, flow_params)

def test_training():
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
    import espalomax as esp
    polynomial_parameters = esp.flow.get_polynomial_parameters()

    from functools import partial
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("CC")
    graph = esp.Graph.from_openff_molecule(molecule)
    model = esp.nn.Parametrization(
        representation=esp.nn.GraphSageModel(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3, out_features=polynomial_parameters),

    )
    nn_params = model.init(jax.random.PRNGKey(2666), graph)

    parameters  = esp.graph.parameters_from_molecule(molecule)
    from jax_md import space
    displacement_fn, shift_fn = space.free()

    from jax_md.mm import mm_energy_fn
    energy_fn, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=parameters,
    )

    def f(x, t, flow_params):
        ff_params = esp.flow.eval_polynomial(t, flow_params)
        f_esp = jax.grad(esp.mm.get_energy, 1)(ff_params, x)
        return f_esp

    def loss(x, nn_params):
        flow_params = model.apply(nn_params, graph)
        print(flow_params)
        y = odeint(f, x, jnp.array([0.0, 1.0]), flow_params)
        u = energy_fn(y)
        return u

    key = jax.random.PRNGKey(2666)
    x = jax.random.normal(key, shape=(8, 3))
    loss(x, nn_params)
