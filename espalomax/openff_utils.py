from openff.toolkit.topology import Molecule
import jax
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

def canonical_featurizer(molecule: Molecule) -> jnp.ndarray:
    def fp(atom):
        from rdkit import Chem
        HYBRIDIZATION_RDKIT = {
            Chem.rdchem.HybridizationType.SP: jnp.array(
                [1, 0, 0, 0, 0],
                dtype=jnp.float32,
            ),
            Chem.rdchem.HybridizationType.SP2: jnp.array(
                [0, 1, 0, 0, 0],
                dtype=jnp.float32,
            ),
            Chem.rdchem.HybridizationType.SP3: jnp.array(
                [0, 0, 1, 0, 0],
                dtype=jnp.float32,
            ),
            Chem.rdchem.HybridizationType.SP3D: jnp.array(
                [0, 0, 0, 1, 0],
                dtype=jnp.float32,
            ),
            Chem.rdchem.HybridizationType.SP3D2: jnp.array(
                [0, 0, 0, 0, 1],
                dtype=jnp.float32,
            ),
            Chem.rdchem.HybridizationType.S: jnp.array(
                [0, 0, 0, 0, 0],
                dtype=jnp.float32,
            ),
        }
        return jnp.concatenate(
            [
                jnp.array(
                    [
                        atom.GetTotalDegree(),
                        atom.GetTotalValence(),
                        atom.GetExplicitValence(),
                        atom.GetFormalCharge(),
                        atom.GetIsAromatic() * 1.0,
                        atom.GetMass(),
                        atom.IsInRingSize(3) * 1.0,
                        atom.IsInRingSize(4) * 1.0,
                        atom.IsInRingSize(5) * 1.0,
                        atom.IsInRingSize(6) * 1.0,
                        atom.IsInRingSize(7) * 1.0,
                        atom.IsInRingSize(8) * 1.0,
                    ],
                    dtype=jnp.float32,
                ),
                HYBRIDIZATION_RDKIT[atom.GetHybridization()],
            ],
            axis=0,
        )

    nodes = [atom.atomic_number for atom in molecule.atoms]
    nodes = jnp.array(nodes)
    nodes = jax.nn.one_hot(nodes, 100)  # TODO: use more features
    nodes_fp = jnp.stack(
        [fp(atom) for atom in molecule.to_rdkit().GetAtoms()], axis=0
    )
    nodes = jnp.concatenate([nodes, nodes_fp], -1)
    return nodes
