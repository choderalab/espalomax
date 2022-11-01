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

def test_nitrogen_gas_bond_energy_consistency():
    from functools import partial
    from openff.toolkit.topology import Molecule
    import jax
    import jax.numpy as jnp
    import jax_md
    import espalomax as esp
    molecule = Molecule.from_smiles("N#N")
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

    assert jnp.allclose(u_esp.flatten(), u_ref.flatten())




# def test_snapshots_consistency():
#     import jax
#     import jax.numpy as jnp
#     from openff.toolkit.topology import Molecule
#     from openff.toolkit.typing.engines.smirnoff import ForceField
#     import espalomax as esp
#     smiles = "N#N"
#     molecule = Molecule.from_smiles(smiles)
#     molecule.assign_partial_charges("zeros")
#     forcefield = ForceField("openff_unconstrained-2.0.0.offxml")
#     system = forcefield.create_openmm_system(
#         molecule.to_topology(),
#         charge_from_molecules=[molecule],
#     )
#
#     from jax_md.mm_utils import parameters_from_openmm_system
#     parameters = parameters_from_openmm_system(system)
#     from jax_md.mm import PeriodicTorsionParameters, HarmonicAngleParameters
#     parameters_new = parameters._replace(
#             periodic_torsion_parameters=PeriodicTorsionParameters(
#                 particles=parameters.periodic_torsion_parameters.particles,
#                 amplitude=jnp.zeros_like(parameters.periodic_torsion_parameters.amplitude),
#                 periodicity=parameters.periodic_torsion_parameters.periodicity,
#                 phase=parameters.periodic_torsion_parameters.phase,
#             ),
#             harmonic_angle_parameters=HarmonicAngleParameters(
#                 particles=parameters.harmonic_angle_parameters.particles,
#                 epsilon=jnp.zeros_like(parameters.harmonic_angle_parameters.epsilon),
#                 length=jnp.zeros_like(parameters.harmonic_angle_parameters.length),
#             ),
#     )
#     molecule.generate_conformers()
#     coordinate = molecule.conformers[0]._value * 0.1
#     coordinate = jnp.array(coordinate)
#
#     from jax_md import space
#     displacement_fn, shift_fn = space.free()
#
#     from jax_md.mm import mm_energy_fn
#     energy_fn, _ = mm_energy_fn(
#         displacement_fn, default_mm_parameters=parameters,
#     )
#
#     energy_fn_new, _ = mm_energy_fn(
#         displacement_fn, default_mm_parameters=parameters_new,
#     )
#
#     from jax_md import simulate
#     temperature = 1.0
#     dt = 1e-3
#     init, update = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, temperature)
#     state = init(jax.random.PRNGKey(2666), coordinate)
#     update = jax.jit(update)
#
#     traj = []
#     for _ in range(100):
#         state = update(state)
#         traj.append(state.position)
#     traj = jnp.stack(traj)
#     u = jax.vmap(energy_fn_new)(traj)
#     u = u - esp.mm.get_nonbonded_energy(molecule, traj)
