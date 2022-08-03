def run():
    from openff.toolkit.typing.engines.smirnoff import ForceField
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("N#N")
    molecule.assign_partial_charges("mmff94")
    forcefield = ForceField("openff_unconstrained-2.0.0.offxml")
    system = forcefield.create_openmm_system(
        molecule.to_topology(),
        charge_from_molecules=[molecule],
    )
    from jax_md.mm_utils import parameters_from_openmm_system
    parameters = parameters_from_openmm_system(system)

    from jax_md import space
    displacement_fn, _ = space.free()

    from jax_md.mm import mm_energy_fn
    energy_fn, _ = mm_energy_fn(
        displacement_fn, parameters,
        space_shape=space.free,
        use_neighbor_list=False,
        box_size=None,
        use_multiplicative_isotropic_cutoff=False,
        use_dsf_coulomb=False,
        neighbor_kwargs={},
    )

if __name__ == "__main__":
    run()
