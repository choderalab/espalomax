import jax
import jax.numpy as jnp
import jax_md

def get_distances(conformations, idxs):
    x0 = conformations[idxs[:, 0]]
    x1 = conformations[idxs[:, 1]]
    x01 = x0 - x1
    return jax_md.space.distance(x01)

def get_angles(conformations, idxs):
    x0 = conformations[idxs[:, 0]]
    x1 = conformations[idxs[:, 1]]
    x2 = conformations[idxs[:, 2]]
    x10 = x0 - x1
    x12 = x2 - x1
    return jnp.acos(
        jax_md.quantity.cosine_angle_between_two_vectors(x10, x_12),
    )

def get_dihedrals(conformations, idxs):
    x0 = conformations[idxs[:, 0]]
    x1 = conformations[idxs[:, 1]]
    x2 = conformations[idxs[:, 2]]
    x3 = conformations[idxs[:, 3]]
    x12 = x2 - x1
    x32 = x2 - x3
    x34 = x4 - x3
    return jax_md.quantity.angle_between_two_half_planes(x12, x32, x34)

def get_bond_energy(conformations, idxs, length, epsilon):
    distances = get_distances(conformations, idxs)
    return jax_md.energy.simple_spring(distances, length, epsilon)

def get_angle_energy(conformations, idxs, length, epsilon):
    angles = get_angles(conformations, idxs)
    return jax_md.energy.simple_spring(angles, length, epsilon)

def get_torsion_energy(conformations, idxs, amplitude, periodicity, phase):
    dihedrals = get_dihedrals(conformations, idxs)
    return jax_md.energy.periodic_torsion(amplitude, periodicity, phase)

def get_energy(
    parameters: Mapping,
    conformations: jnp.ndarray:
):
    """Compute the energy given a set of parameters and conformations.

    Parameters
    ----------
    parameters : Mapping
        esp.nn generated parameters
    conformations : jnp.ndarray
        set of conformations
    """
    raise NotImplementedError
    
