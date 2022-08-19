from typing import Mapping, Optional
from functools import partial
from openff.toolkit.topology import Molecule
import jax
import jax.numpy as jnp
import jax_md
from jax_md.mm import (
    MMEnergyFnParameters,
    HarmonicBondParameters,
    HarmonicAngleParameters,
    PeriodicTorsionParameters,
)

from .nn import BOND_PHASES, ANGLE_PHASES
from .graph import Graph, parameters_from_molecule

def get_distances(conformations, idxs):
    x0 = conformations[..., idxs[:, 0], :]
    x1 = conformations[..., idxs[:, 1], :]
    x01 = x0 - x1
    return jax_md.space.distance(x01)

def get_angles(conformations, idxs):
    x0 = conformations[..., idxs[:, 0], :]
    x1 = conformations[..., idxs[:, 1], :]
    x2 = conformations[..., idxs[:, 2], :]
    x10 = x0 - x1
    x12 = x2 - x1

    fn = jax_md.quantity.cosine_angle_between_two_vectors
    for _ in range(len(x0.shape)-1):
        fn = jax.vmap(fn)
    return fn(x10, x12)

def get_dihedrals(conformations, idxs):
    x1 = conformations[..., idxs[:, 0], :]
    x2 = conformations[..., idxs[:, 1], :]
    x3 = conformations[..., idxs[:, 2], :]
    x4 = conformations[..., idxs[:, 3], :]
    x12 = x2 - x1
    x32 = x2 - x3
    x34 = x4 - x3

    fn = jax_md.quantity.angle_between_two_half_planes
    for _ in range(len(x1.shape)-1):
        fn = jax.vmap(fn)
    return fn(x12, x32, x34)

def linear_mixture_energy(x, coefficients, phases):
    b0, b1 = phases
    k0, k1 = jnp.exp(coefficients[..., 0]), jnp.exp(coefficients[..., 1])
    return k0 * (x - b0) ** 2 + k1 * (x - b1) ** 2

def get_bond_energy(conformations, idxs, coefficients):
    distances = get_distances(conformations, idxs)
    return linear_mixture_energy(distances, coefficients, BOND_PHASES)

def get_angle_energy(conformations, idxs, coefficients):
    angles = get_angles(conformations, idxs)
    return linear_mixture_energy(angles, coefficients, ANGLE_PHASES)

def get_torsion_energy(conformations, idxs, amplitude, periodicity, phase):
    dihedrals = get_dihedrals(conformations, idxs)
    return jax_md.energy.periodic_torsion(dihedrals, amplitude, periodicity, phase)

def get_energy(
    parameters: Mapping,
    conformations: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    batch_size: Optional[int] = None,
):
    """Compute the energy given a set of parameters and conformations.

    Parameters
    ----------
    parameters : Mapping
        esp.nn generated parameters
    conformations : jnp.ndarray
        set of conformations
    """
    proper_idxs = parameters["proper"]["idxs"]
    proper_k = parameters["proper"]["k"]
    improper_idxs = parameters["improper"]["idxs"]
    improper_k = parameters["improper"]["k"]

    if jnp.size(proper_idxs) == 0:
        proper_idxs = jnp.zeros((0, 4), jnp.int32)
        proper_k = jnp.zeros((0, 6), jnp.float32)

    if jnp.size(improper_idxs) == 0:
        improper_idxs = jnp.zeros((0, 4), jnp.int32)
        improper_k = jnp.zeros((0, 6), jnp.float32)

    bond_energy = get_bond_energy(
        conformations,
        idxs=parameters["bond"]["idxs"],
        coefficients=parameters["bond"]["coefficients"],
    )

    angle_energy = get_angle_energy(
        conformations,
        idxs=parameters["angle"]["idxs"],
        coefficients=parameters["angle"]["coefficients"],
    )

    proper_energy = get_torsion_energy(
        conformations,
        idxs=jnp.repeat(proper_idxs, 6, 0),
        amplitude=proper_k.flatten(),
        periodicity=jnp.tile(
            jnp.arange(1, 7),
            len(parameters["proper"]["idxs"])
        ),
        phase=jnp.zeros(
            6 * len(parameters["proper"]["idxs"])
        ),
    )

    improper_energy = get_torsion_energy(
        conformations,
        idxs=jnp.repeat(improper_idxs, 6, 0),
        amplitude=improper_k.flatten(),
        periodicity=jnp.tile(
            jnp.arange(1, 7),
            len(parameters["improper"]["idxs"])
        ),
        phase=jnp.zeros(
            6 * len(parameters["improper"]["idxs"])
        ),
    )

    if mask is None:
        return bond_energy.sum(-1) + angle_energy.sum(-1) + proper_energy.sum(-1) + improper_energy.sum(-1)
    else:
        bond_energy = jax.ops.segment_sum(bond_energy.swapaxes(0, -1), mask["bond"]["mask"], num_segments=batch_size)
        angle_energy = jax.ops.segment_sum(angle_energy.swapaxes(0, -1), mask["angle"]["mask"], num_segments=batch_size)
        proper_energy = jax.ops.segment_sum(proper_energy.swapaxes(0, -1), jnp.repeat(mask["proper"]["mask"], 6, 0), num_segments=batch_size)
        improper_energy = jax.ops.segment_sum(improper_energy.swapaxes(0, -1), jnp.repeat(mask["improper"]["mask"], 6, 0), num_segments=batch_size)

    return bond_energy + angle_energy + proper_energy + improper_energy

def get_nonbonded_energy(
    molecule: Molecule,
    coordinates: jnp.ndarray,
):
    parameters = parameters_from_molecule(molecule)
    parameters = parameters._replace(
        harmonic_bond_parameters=HarmonicBondParameters(
            particles=parameters.harmonic_bond_parameters.particles,
            epsilon=jnp.zeros_like(parameters.harmonic_bond_parameters.epsilon),
            length=jnp.zeros_like(parameters.harmonic_bond_parameters.length),
        ),
        harmonic_angle_parameters=HarmonicAngleParameters(
            particles=parameters.harmonic_angle_parameters.particles,
            epsilon=jnp.zeros_like(parameters.harmonic_angle_parameters.epsilon),
            length=jnp.zeros_like(parameters.harmonic_angle_parameters.length),
        ),
        periodic_torsion_parameters=PeriodicTorsionParameters(
            particles=parameters.periodic_torsion_parameters.particles,
            amplitude=jnp.zeros_like(parameters.periodic_torsion_parameters.amplitude),
            periodicity=parameters.periodic_torsion_parameters.periodicity,
            phase=parameters.periodic_torsion_parameters.phase,
        ),
    )

    from jax_md import space
    from jax_md.mm import mm_energy_fn
    displacement_fn, _ = space.free()

    energy_fn, _ = mm_energy_fn(
        displacement_fn, parameters,
        space_shape=space.free,
        use_neighbor_list=False,
        box_size=None,
        use_multiplicative_isotropic_cutoff=False,
        use_dsf_coulomb=False,
        neighbor_kwargs={},
    )

    energy_fn = partial(energy_fn, parameters=parameters)
    energy_fn = jax.vmap(energy_fn, 0)
    u = energy_fn(coordinates)
    return u
