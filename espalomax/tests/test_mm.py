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
    assert onp.allclose(coefficients, coefficients1, rtol=1e-2, atol=1e-2)

def test_nitrogen_gas_bond_energy_consistency():
    from functools import partial
    from openff.toolkit.topology import Molecule
    import jax
    import jax.numpy as jnp
    import jax_md
    import espalomax as esp
    molecule = Molecule.from_smiles("C")
    base_parameters = esp.graph.parameters_from_molecule(molecule)
    graph = esp.Graph.from_openff_molecule(molecule)
    coordinates = jax.random.normal(jax.random.PRNGKey(2666), shape=(8, 2, 3))

    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3),
    )

    nn_params = model.init(jax.random.PRNGKey(2666), graph)
    ff_params_esp = model.apply(nn_params, graph)
    ff_params_esp["angle"]["coefficients"] = jnp.zeros((0, 2), jnp.float32)

    get_energy = jax.vmap(esp.mm.get_energy, (None, 0))
    u_esp = get_energy(ff_params_esp, coordinates)

    k, b = esp.mm.linear_mixture_to_original(
        ff_params_esp["bond"]["coefficients"],
        esp.mm.BOND_PHASES,
    )

    distances = esp.mm.get_distances(coordinates, ff_params_esp["bond"]["idxs"])
    u_ref = k * (distances - b) ** 2

    u_esp = u_esp - u_esp.mean()
    u_ref = u_ref - u_ref.mean()

    assert jnp.allclose(u_esp.flatten(), u_ref.flatten(), rtol=1e-2, atol=1e-2)


def test_snapshots_consistency():
    import jax
    import jax.numpy as jnp
    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import ForceField
    import espalomax as esp
    # smiles = "N#N"
    smiles = "CCCC"
    molecule = Molecule.from_smiles(smiles)
    parameters  = esp.graph.parameters_from_molecule(molecule)

    from jax_md.mm import PeriodicTorsionParameters, HarmonicAngleParameters, HarmonicBondParameters

    parameters = parameters._replace(
        # harmonic_bond_parameters=HarmonicBondParameters(
        #     particles=parameters.harmonic_bond_parameters.particles,
        #     epsilon=4.0 * jnp.ones_like(parameters.harmonic_bond_parameters.epsilon),
        #     length=0.5 * jnp.ones_like(parameters.harmonic_bond_parameters.length),
        # ),
        # harmonic_angle_parameters=HarmonicAngleParameters(
        #     particles=jnp.zeros((0, 3), dtype=jnp.int32),
        #     epsilon=jnp.zeros((0, ), dtype=jnp.float32),
        #     length=jnp.zeros((0, ), dtype=jnp.float32),
        # ),
        periodic_torsion_parameters=PeriodicTorsionParameters(
            particles=jnp.zeros((0, 4), dtype=jnp.int32),
            phase=jnp.zeros((0, ), dtype=jnp.float32),
            periodicity=jnp.zeros((0, ), dtype=jnp.float32),
            amplitude=jnp.zeros((0, ), dtype=jnp.float32),
        ),
    )

    parameters_new = parameters._replace(
            harmonic_bond_parameters=HarmonicBondParameters(
                particles=parameters.harmonic_bond_parameters.particles,
                epsilon=jnp.zeros_like(parameters.harmonic_bond_parameters.epsilon),
                length=parameters.harmonic_bond_parameters.length,
            ),
            harmonic_angle_parameters=HarmonicAngleParameters(
                particles=parameters.harmonic_angle_parameters.particles,
                epsilon=jnp.zeros_like(parameters.harmonic_angle_parameters.epsilon),
                length=parameters.harmonic_angle_parameters.length,
            ),
    )
    molecule.generate_conformers()
    coordinate = molecule.conformers[0]._value * 0.1
    coordinate = jnp.array(coordinate)

    from jax_md import space
    displacement_fn, shift_fn = space.free()

    from jax_md.mm import mm_energy_fn
    energy_fn, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=parameters,
    )

    energy_fn_new, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=parameters_new,
    )

    from jax_md import simulate
    temperature = 1.0
    dt = 1e-3
    init, update = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, temperature)
    state = init(jax.random.PRNGKey(2666), coordinate)
    update = jax.jit(update)

    traj = []
    for _ in range(100):
        state = update(state)
        traj.append(state.position)
    traj = jnp.stack(traj)


    # traj = jax.random.normal(jax.random.PRNGKey(2666), shape=(100, 8, 3))
    u = jax.vmap(energy_fn)(traj) - jax.vmap(energy_fn_new)(traj)
    # u = u / 627.5

    model = esp.nn.Parametrization(
        representation=esp.nn.GraphSageModel(32, 3),
        janossy_pooling=esp.nn.JanossyPooling(32, 4),
    )

    g = esp.Graph.from_openff_molecule(molecule)

    def get_loss(nn_params, g, x, u):
        ff_params = model.apply(nn_params, g)
        u_hat = esp.mm.get_energy(ff_params, x, terms=["bond", "angle"])
        u_hat = u_hat - u_hat.mean()
        u = u - u.mean()
        return ((u - u_hat) ** 2).mean()

    @jax.jit
    def step(state, g, x, u):
        nn_params = state.params
        loss, grads = jax.value_and_grad(get_loss)(nn_params, g, x, u)
        jax.debug.print("{x}", x=loss)
        state = state.apply_gradients(grads=grads)
        return state

    import optax
    optimizer = optax.adamw(learning_rate=1e-5, weight_decay=1e-10)

    nn_params = model.init(jax.random.PRNGKey(2667), g)
    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
         apply_fn=model.apply, params=nn_params, tx=optimizer,
    )

    # from matplotlib import pyplot as plt
    # plt.scatter(u_hat, u)
    # plt.show()


    import random
    for idx_batch in range(1000000):
            state = step(state, g, traj, u)

test_snapshots_consistency()
