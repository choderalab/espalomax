from jax_md.mm import *

def mm_energy_components_fn(displacement_fn : DisplacementFn,
                 parameters : MMEnergyFnParameters,
                 space_shape : Union[space.free, space.periodic, space.periodic_general] = space.periodic,
                 use_neighbor_list : Optional[bool] = True,
                 box_size: Optional[Box] = 1.,
                 use_multiplicative_isotropic_cutoff: Optional[bool]=True,
                 use_dsf_coulomb: Optional[bool]=True,
                 neighbor_kwargs: Optional[Dict[str, Any]]=None,
                 multiplicative_isotropic_cutoff_kwargs: Optional[Dict[str, Any]]={},
                 **unused_kwargs,
                 ) -> Union[EnergyFn, partition.NeighborListFns]:
  """
  generator of a canonical molecular mechanics-like `EnergyFn`;
  TODO :
    - render `nonbonded_exception_parameters.particles` static (requires changing `custom_mask_fn` handler)
    - retrieve standard nonbonded energy (already coded)
    - retrieve nonbonded energy per particle
    - render `.particles` parameters static (may affect jit compilation speeds)
  Args:
    displacement_fn: A `DisplacementFn`.
    parameters: An `MMEnergyFnParameters` containing all of the parameters of the model;
      While these are dynamic, `mm_energy_fn` does effectively make some of these static, namely the `particles` parameter
    space_shape : A generalized `jax_md.space`
    use_neighbor_list : whether to use a neighbor list for `NonbondedParameters`
    box_size : size of box for `space.periodic` or `space.periodic_general`; omitted for `space.free`
  Returns:
    An `EnergyFn` taking positions R (an ndarray of shape [n_particles, 3]), parameters,
      (optionally) a `NeighborList`, and optional kwargs
    A `neighbor_fn` for allocating and updating a neighbor_list
  Example (vacuum from `openmm`):
  >>> pdb = app.PDBFile('alanine-dipeptide-explicit.pdb')
  >>> ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
  >>> mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False, removeCMMotion=False)
  >>> model = Modeller(pdb.topology, pdb.positions)
  >>> model.deleteWater()
  >>> mmSystem = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
  >>> context = openmm.Context(mmSystem, openmm.VerletIntegrator(1.*unit.femtoseconds))
  >>> context.setPositions(model.getPositions())
  >>> omm_state = context.getState(getEnergy=True, getPositions=True)
  >>> positions = jnp.array(omm_state.getPositions(asNumpy=True).value_in_unit_system(unit.md_unit_system))
  >>> energy = omm_state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
  >>> from jax_md import mm_utils
  >>> params = mm_utils.parameters_from_openmm_system(mmSystem)
  >>> displacement_fn, shift_fn = space.free()
  >>> energy_fn, neighbor_list = mm_energy_fn(displacement_fn=displacement_fn,
                                           parameters = params,
                                           space_shape=space.free,
                                           use_neighbor_list=False,
                                           box_size=None,
                                           use_multiplicative_isotropic_cutoff=False,
                                           use_dsf_coulomb=False,
                                           neighbor_kwargs={},
                                           )
  >>> out_energy = energy_fn(positions, parameters = params) # retrieve potential energy in units of `openmm.unit.md_unit_system` (kJ/mol)
  """
  check_support(space_shape, use_neighbor_list, box_size)

  # bonded energy fns
  bond_fns = get_bond_fns(displacement_fn) # get geometry handlers dict
  parameter_template = check_parameters(parameters) # just make sure that parameters
  for _key in bond_fns:
    bond_fns[_key] = util.merge_dicts(bond_fns[_key], parameter_template[_key])
  bonded_energy_fns = {}
  for parameter_field in parameters._fields:
    if parameter_field in list(bond_fns.keys()): # then it is bonded
      mapped_bonded_energy_fn = smap.bond(
                                     displacement_or_metric=displacement_fn,
                                     **bond_fns[parameter_field], # `geometry_handler_fn` and `fn`
                                     )
      bonded_energy_fns[parameter_field] = mapped_bonded_energy_fn
    elif parameter_field in [camel_to_snake(_entry) for _entry in CANONICAL_MM_NONBONDED_PARAMETER_NAMES]: # nonbonded
      nonbonded_parameters = getattr(parameters, parameter_field)
      are_nonbonded_exception_parameters = True if 'nonbonded_exception_parameters' in parameters._fields else False
      if are_nonbonded_exception_parameters: # handle custom nonbonded mask
        n_particles=nonbonded_parameters.charge.shape[0] # query the number of particles
        padded_exception_array = query_idx_in_pair_exceptions(indices=jnp.arange(n_particles), pair_exceptions=getattr(getattr(parameters, 'nonbonded_exception_parameters'), 'particles'))
        custom_mask_fn = nonbonded_exception_mask_fn(n_particles=n_particles, padded_exception_array=padded_exception_array)
        neighbor_kwargs = util.merge_dicts({'custom_mask_fn': custom_mask_fn}, neighbor_kwargs)
      nonbonded_energy_fn, neighbor_fn = nonbonded_neighbor_list(displacement_or_metric=displacement_fn,
                             nonbonded_parameters=getattr(parameters, parameter_field),
                             use_neighbor_list=use_neighbor_list,
                             use_multiplicative_isotropic_cutoff=use_multiplicative_isotropic_cutoff,
                             use_dsf_coulomb=use_dsf_coulomb,
                             multiplicative_isotropic_cutoff_kwargs=multiplicative_isotropic_cutoff_kwargs,
                             particle_exception_indices=parameters.nonbonded_exception_parameters.particles if are_nonbonded_exception_parameters else None,
                             neighbor_kwargs=neighbor_kwargs)
    else:
      raise NotImplementedError(f"""parameter name {parameter_field} is not currently supported by
      `CANONICAL_MM_BOND_PARAMETER_NAMES` or `CANONICAL_MM_NONBONDED_PARAMETER_NAMES`""")

  def bond_handler(parameters, **unused_kwargs):
    """a simple function to easily `tree_map` bond functions"""
    bonds = {_key: getattr(parameters, _key).particles for _key in parameters._fields if _key in bond_fns.keys()}
    bond_types = {_key: getattr(parameters, _key)._replace(particles=None)._asdict() for _key in bonds.keys()}
    return bonds, bond_types

  def nonbonded_handler(parameters, **unused_kwargs) -> Dict:
    if 'nonbonded_parameters' in parameters._fields:
      return parameters.nonbonded_parameters._asdict()
    else:
      return {}

  def energy_fn(R: Array, **dynamic_kwargs) -> Array:
    bonds, bond_types = bond_handler(**dynamic_kwargs)
    bonded_energies = jax.tree_util.tree_map(lambda _f, _bonds, _bond_types : _f(R, _bonds, _bond_types),
                                             bonded_energy_fns, bonds, bond_types)
    accum = accum + util.high_precision_sum(jnp.array(list(bonded_energies.values())))

    # nonbonded
    nonbonded_parameters = nonbonded_handler(**dynamic_kwargs)
    nonbonded_energy = nonbonded_energy_fn(R, nonbonded_parameters_dict=nonbonded_parameters)
    accum = accum + nonbonded_energy # handle if/not in
    return accum

  return energy_fn, neighbor_fn
