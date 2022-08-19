from openff.toolkit.topology import Molecule
import jax.numpy as jnp

def get_bond_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    bond_idxs = jnp.array(
        [[bond.atom1_index, bond.atom2_index] for bond in molecule.bonds],
        dtype=jnp.int32,
    )
    if len(bond_idxs) == 0:
        bond_idxs = jnp.zeros((0, 2), jnp.int32)
    return bond_idxs

def get_angle_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    angle_idxs = jnp.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in angle])
                for angle in molecule.angles
            ]
        ),
        dtype=jnp.int32,
    )
    if len(angle_idxs) == 0:
        angle_idxs = jnp.zeros((0, 3), jnp.int32)
    return angle_idxs

def get_proper_torsion_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    proper_torsion_idxs = jnp.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in proper])
                for proper in molecule.propers
            ]
        ),
        dtype=jnp.int32,
    )
    if len(proper_torsion_idxs) == 0:
        proper_torsion_idxs = jnp.zeros((0, 4), jnp.int32)
    return proper_torsion_idxs

def get_improper_torsion_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    improper_smarts = '[*:2]~[X3:1](~[*:3])~[*:4]'
    ## For smirnoff ordering, we only want to find the unique combinations
    ##  of atoms forming impropers so we can permute them the way we want
    mol_idxs = molecule.chemical_environment_matches(improper_smarts,
        unique=True)

    ## Get all ccw orderings
    # feels like there should be some good way to do this with itertools...
    idx_permuts = []
    for c, *other_atoms in mol_idxs:
        for i in range(3):
            idx = [c]
            for j in range(3):
                idx.append(other_atoms[(i+j)%3])
            idx_permuts.append(tuple(idx))

    improper_torsion_idxs = jnp.array(idx_permuts, dtype=jnp.int32)
    if len(improper_torsion_idxs) == 0:
        improper_torsion_idxs = jnp.zeros((0, 4), jnp.int32)
    return improper_torsion_idxs

def get_nonbonded_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    import networkx as nx
    adj = nx.adjacency_matrix(molecule.to_networkx()).todense()
    a_ = jnp.array(adj)

    nonbonded_idxs = jnp.stack(
        jnp.where(
            jnp.equal(a_ + a_ @ a_ + a_ @ a_ @ a_, 0.0),
        ),
        axis=-1,
    )
    if len(nonbonded_idxs) == 0:
        nonbonded_idxs = jnp.zeros((0, 2), jnp.int32)
    return nonbonded_idxs

def get_onefour_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    import networkx as nx
    adj = nx.adjacency_matrix(molecule.to_networkx()).todense()
    a_ = jnp.array(adj)

    onefour_idxs = jnp.stack(
        jnp.where(
            jnp.equal(a_ + a_ @ a_, 0.0) * jnp.greater(a_ @ a_ @ a_, 0.0),
        ),
        axis=-1,
    )
    if len(onefour_idxs) == 0:
        onefour_idxs = jnp.zeros((0, 2), jnp.int32)
    return onefour_idxs
