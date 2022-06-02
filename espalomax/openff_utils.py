from openff.toolkit.topology import Molecule
import jax.numpy as jnp

def get_bond_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    bond_idxs = jnp.array(
        [[bond.atom1_index, bond.atom2_index] for bond in molecule.bonds]
    )
    return bond_idxs

def get_angle_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    angle_idxs = jnp.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in angle])
                for angle in molecule.angles
            ]
        )
    )

    return angle_idxs

def get_proper_torsion_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    proper_torsion_idxs = jnp.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in proper])
                for proper in molecule.propers
            ]
        )
    )

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

    improper_torsion_idxs = jnp.array(idx_permuts)
    return improper_torsion_idxs

def get_nonbonded_idxs_from_molecule(molecule: Molecule) -> jnp.ndarray:
    import networkx as nx
    adj = nx.adjacency_matrix(molecule.to_networkx()).todense()
    a_ = jnp.array(adj)

    nonbonded_idxs = jnp.stack(
        jnp.where(
            jnp.equal(a_ + a_ @ a_ + a_ @ a_ @ a_, 0.0)
        ),
        axis=-1,
    )
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

    return onefour_idxs
