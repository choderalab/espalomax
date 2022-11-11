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

    from diffrax import diffeqsolve, ODETerm, Dopri5

    def f(t, x, flow_params):
        ff_params = esp.flow.eval_polynomial(t, flow_params)
        f_esp = jax.vmap(jax.grad(esp.mm.get_energy, 1), (None, 0))(ff_params, x)
        return f_esp

    term = ODETerm(f)
    solver = Dopri5()

    def loss(nn_params, x):
        flow_params = model.apply(nn_params, graph)
        flow_params = esp.flow.constraint_polynomial_parameters(flow_params)
        y = diffeqsolve(term, solver, args=flow_params, t0=0.0, t1=1.0,  dt0=0.1, y0=x, max_steps=1000).ys[-1]
        u = jax.vmap(energy_fn)(y).mean()
        jax.debug.print("{x}", x=u)
        return u

    @jax.jit
    def step(state, x):
        grads = jax.grad(loss)(state.params, x)
        state = state.apply_gradients(grads=grads)
        return state

    
    import optax
    tx = optax.adamw(1e-5)
    from flax.training.train_state import TrainState
    state = TrainState.create(apply_fn=model.apply, params=nn_params, tx=tx) 
    key = jax.random.PRNGKey(2666)
    for _ in range(10000):
        this_key, key = jax.random.split(key)
        x = jax.random.normal(key, shape=(10, 8, 3))
        state = step(state, x)

test_training()
