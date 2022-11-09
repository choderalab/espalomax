import pytest
#
# def test_hardcode_energy():
#     from openff.toolkit.topology import Molecule
#     import jax
#     import jax.numpy as jnp
#     import espalomax as esp
#
#     molecule = Molecule.from_smiles("C=C")
#     graph = esp.Graph.from_openff_molecule(molecule)
#
#     model = esp.nn.Parametrization(
#         representation=esp.nn.GraphAttentionNetwork(8, 3),
#         janossy_pooling=esp.nn.JanossyPooling(8, 3),
#     )
#
#     nn_params = model.init(jax.random.PRNGKey(2666), graph)
#     ff_params = model.apply(nn_params, graph)
#     get_energy = jax.vmap(esp.mm.get_energy, (None, 0))
#
#     u = get_energy(
#         ff_params,
#         jax.random.normal(jax.random.PRNGKey(2666), shape=(2, 6, 3))
#     )
#
#     assert u.shape == (2, )
#
#
# def test_batch_energy():
#     from openff.toolkit.topology import Molecule
#     import jax
#     import jax.numpy as jnp
#     import espalomax as esp
#
#     g0 = esp.Graph.from_openff_molecule(Molecule.from_smiles("C=C"))
#     g1 = esp.Graph.from_openff_molecule(Molecule.from_smiles("C=CC"))
#     g = esp.graph.batch((g0, g1))
#     heteromask = esp.graph.heteromask(g)
#
#     model = esp.nn.Parametrization(
#         representation=esp.nn.GraphAttentionNetwork(8, 3),
#         janossy_pooling=esp.nn.JanossyPooling(8, 3),
#     )
#
#     nn_params = model.init(jax.random.PRNGKey(2666), g)
#     ff_params = model.apply(nn_params, g)
#     # get_energy = jax.vmap(esp.mm.get_energy, (None, 0, None))
#
#     u = esp.mm.get_energy(
#         ff_params,
#         jax.random.normal(jax.random.PRNGKey(2666), shape=(4, g.n_atoms, 3)),
#         heteromask,
#     )
#
#     assert u.shape == (2, 4)



#
# def test_nonbonded_energy():
#     from openff.toolkit.topology import Molecule
#     import jax
#     import jax.numpy as jnp
#     import espalomax as esp
#
#     molecule = Molecule.from_smiles("C=C")
#     graph = esp.Graph.from_openff_molecule(molecule)
#
#     x = jax.random.normal(jax.random.PRNGKey(2666), shape=(2, 6, 3))
#     u = esp.mm.get_nonbonded_energy(molecule, x)
#     assert u.shape == (2, )
#
# def test_jax_md_energy():
#     from openff.toolkit.topology import Molecule
#     import jax
#     import jax.numpy as jnp
#     import jax_md
#     import espalomax as esp
#     molecule = Molecule.from_smiles("C=C")
#     base_parameters = esp.graph.parameters_from_molecule(molecule)
#     graph = esp.Graph.from_openff_molecule(molecule)
#
#     model = esp.nn.Parametrization(
#         representation=esp.nn.GraphAttentionNetwork(8, 3),
#         janossy_pooling=esp.nn.JanossyPooling(8, 3),
#     )
#
#     nn_params = model.init(jax.random.PRNGKey(2666), graph)
#     ff_params = model.apply(nn_params, graph)
#     ff_params = esp.nn.to_jaxmd_mm_energy_fn_parameters(ff_params, base_parameters)
#
#     from jax_md import space
#     displacement_fn, shift_fn = space.free()
#
#     from jax_md.mm import mm_energy_fn
#     energy_fn, neighbor_fn = mm_energy_fn(
#         displacement_fn, ff_params,
#         space_shape=space.free,
#         use_neighbor_list=False,
#         box_size=None,
#         use_multiplicative_isotropic_cutoff=False,
#         use_dsf_coulomb=False,
#         neighbor_kwargs={},
#     )
#
#     u = energy_fn(
#         jax.random.normal(key=jax.random.PRNGKey(2666), shape=(6, 3)),
#         parameters=ff_params
#     )

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

    from jax_md.mm import HarmonicBondParameters, HarmonicAngleParameters
    ff_params_jaxmd = esp.mm.to_jaxmd_mm_energy_fn_parameters(ff_params_esp, base_parameters)
    ff_params_jaxmd_without_bond = ff_params_jaxmd._replace(
        harmonic_bond_parameters=HarmonicBondParameters(
            particles=ff_params_jaxmd.harmonic_bond_parameters.particles,
            epsilon=jnp.zeros_like(ff_params_jaxmd.harmonic_bond_parameters.epsilon),
            length=jnp.zeros_like(ff_params_jaxmd.harmonic_bond_parameters.length),
        ),
    )

    from jax_md import space
    displacement_fn, shift_fn = space.free()

    from jax_md.mm import mm_energy_fn
    energy_fn, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=ff_params_jaxmd,
    )

    energy_fn_without_bond, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=ff_params_jaxmd_without_bond,
    )

    u_jax_md = jax.vmap(energy_fn, 0)(coordinates)
    u_jax_md_without_bond = jax.vmap(energy_fn_without_bond, 0)(coordinates)
    u_jax_md = u_jax_md - u_jax_md_without_bond

    get_energy = jax.vmap(esp.mm.get_energy, (None, 0))
    u_esp = get_energy(ff_params_esp, coordinates)

    u_jax_md = u_jax_md - u_jax_md.mean(0)
    u_esp = u_esp - u_esp.mean(0)

    assert jnp.allclose(u_jax_md, u_esp, rtol=1e-2, atol=1e-2)

def test_methane_angle_energy_consistency():
    from functools import partial
    from openff.toolkit.topology import Molecule
    import jax
    import jax.numpy as jnp
    import jax_md
    import espalomax as esp
    molecule = Molecule.from_smiles("C")
    molecule.generate_conformers(n_conformers=2)
    base_parameters = esp.graph.parameters_from_molecule(molecule)
    graph = esp.Graph.from_openff_molecule(molecule)
    # coordinates = jnp.stack([conformer._value for conformer in molecule.conformers])
    coordinates = jax.random.normal(jax.random.PRNGKey(2666), shape=(8, 5, 3))
    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3),
    )

    nn_params = model.init(jax.random.PRNGKey(2666), graph)
    ff_params_esp = model.apply(nn_params, graph)

    from jax_md.mm import HarmonicBondParameters, HarmonicAngleParameters
    ff_params_jaxmd = esp.mm.to_jaxmd_mm_energy_fn_parameters(ff_params_esp, base_parameters)
    ff_params_jaxmd_without_angle = ff_params_jaxmd._replace(
        harmonic_angle_parameters=HarmonicAngleParameters(
            particles=ff_params_jaxmd.harmonic_angle_parameters.particles,
            epsilon=jnp.zeros_like(ff_params_jaxmd.harmonic_angle_parameters.epsilon),
            length=jnp.zeros_like(ff_params_jaxmd.harmonic_angle_parameters.length),
        ),
    )

    ff_params_esp["bond"]["coefficients"] = -1e7 * jnp.ones_like(ff_params_esp["bond"]["coefficients"])
    ff_params_esp["proper"]["k"] = jnp.zeros_like(ff_params_esp["proper"]["k"])
    ff_params_esp["improper"]["k"] = jnp.zeros_like(ff_params_esp["improper"]["k"])

    from jax_md import space
    displacement_fn, shift_fn = space.free()

    from jax_md.mm import mm_energy_fn
    energy_fn, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=ff_params_jaxmd,
    )
    energy_fn(coordinates[0])

    energy_fn_without_angle, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=ff_params_jaxmd_without_angle,
    )

    u_jax_md = jax.vmap(energy_fn, 0)(coordinates)
    u_jax_md_without_angle = jax.vmap(energy_fn_without_angle, 0)(coordinates)
    u_jax_md = u_jax_md - u_jax_md_without_angle

    get_energy = jax.vmap(esp.mm.get_energy, (None, 0))
    u_esp = get_energy(ff_params_esp, coordinates)

    u_jax_md = u_jax_md - u_jax_md.mean(0)
    u_esp = u_esp - u_esp.mean(0)
    assert jnp.allclose(u_jax_md, u_esp, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "smiles", [
        idx * "C" for idx in range(1, 10)
    ],
)
def test_all_bonded_energy(smiles):
    from functools import partial
    from openff.toolkit.topology import Molecule
    import jax
    import jax.numpy as jnp
    import jax_md
    import espalomax as esp
    molecule = Molecule.from_smiles(smiles)
    base_parameters = esp.graph.parameters_from_molecule(molecule)
    graph = esp.Graph.from_openff_molecule(molecule)
    coordinates = jax.random.normal(jax.random.PRNGKey(2666), shape=(8, graph.n_atoms, 3))
    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3),
    )

    nn_params = model.init(jax.random.PRNGKey(2666), graph)
    ff_params_esp = model.apply(nn_params, graph)

    from jax_md.mm import (
        HarmonicBondParameters,
        HarmonicAngleParameters,
        PeriodicTorsionParameters,
    )

    ff_params_jaxmd = esp.mm.to_jaxmd_mm_energy_fn_parameters(ff_params_esp, base_parameters)
    ff_params_jaxmd_without_bonded = ff_params_jaxmd._replace(
        harmonic_bond_parameters=HarmonicBondParameters(
            particles=ff_params_jaxmd.harmonic_bond_parameters.particles,
            epsilon=jnp.zeros_like(ff_params_jaxmd.harmonic_bond_parameters.epsilon),
            length=jnp.zeros_like(ff_params_jaxmd.harmonic_bond_parameters.length),
        ),
        harmonic_angle_parameters=HarmonicAngleParameters(
            particles=ff_params_jaxmd.harmonic_angle_parameters.particles,
            epsilon=jnp.zeros_like(ff_params_jaxmd.harmonic_angle_parameters.epsilon),
            length=jnp.zeros_like(ff_params_jaxmd.harmonic_angle_parameters.length),
        ),
        periodic_torsion_parameters=PeriodicTorsionParameters(
            particles=ff_params_jaxmd.periodic_torsion_parameters.particles,
            amplitude=jnp.zeros_like(ff_params_jaxmd.periodic_torsion_parameters.amplitude),
            periodicity=ff_params_jaxmd.periodic_torsion_parameters.periodicity,
            phase=ff_params_jaxmd.periodic_torsion_parameters.phase,
        ),
    )

    from jax_md import space
    displacement_fn, shift_fn = space.free()

    from jax_md.mm import mm_energy_fn
    energy_fn, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=ff_params_jaxmd,
    )

    energy_fn_without_bonded, _ = mm_energy_fn(
        displacement_fn, default_mm_parameters=ff_params_jaxmd_without_bonded,
    )

    u_jax_md = jax.vmap(energy_fn, 0)(coordinates)
    u_jax_md_without_angle = jax.vmap(energy_fn_without_bonded, 0)(coordinates)
    u_jax_md = u_jax_md - u_jax_md_without_angle

    get_energy = jax.vmap(esp.mm.get_energy, (None, 0))
    u_esp = get_energy(ff_params_esp, coordinates)

    u_jax_md = u_jax_md - u_jax_md.mean(0)
    u_esp = u_esp - u_esp.mean(0)
    assert jnp.allclose(u_jax_md, u_esp, rtol=1e-2, atol=1e-2)
