from typing import NamedTuple, DefaultDict
from openff.toolkit.topology import Molecule
from jraph import GraphsTuple
import jax
import jax.numpy as jnp

from typing import DefaultDict

class Heterograph(DefaultDict):
    def __init__(self, *args, **kwargs):
        super(Heterograph, self).__init__(*args, **kwargs)
        self.default_factory = lambda: DefaultDict(lambda: None)

class Graph(NamedTuple):
    homograph: GraphsTuple
    heterograph: Heterograph

    @staticmethod
    def homograph_from_openff_molecule(molecule: Molecule) -> GraphsTuple:
        # get nodes
        nodes = [atom.atomic_number for atom in molecule.atoms]
        nodes = jnp.array(nodes)
        nodes = jax.nn.one_hot(nodes, 118) # TODO: use more features

        # get bonds
        senders = []
        receivers = []
        for bond in molecule.bonds:
            senders.append(bond.atom1_index)
            receivers.append(bond.atom2_index)
            senders.append(bond.atom2_index)
            receivers.append(bond.atom1_index)
        senders = jnp.array(senders)
        receivers = jnp.array(receivers)

        # count
        n_node = len(nodes)
        n_edge = len(senders)

        return GraphsTuple(
            nodes=nodes,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            edges=None,
            globals=None,
        )

    @staticmethod
    def heterograph_from_openff_molecule(molecule: Molecule) -> Heterograph:
        from .openff_utils import (
            get_bond_idxs_from_molecule,
            get_angle_idxs_from_molecule,
            get_proper_torsion_idxs_from_molecule,
            get_improper_torsion_idxs_from_molecule,
            get_nonbonded_idxs_from_molecule,
            get_onefour_idxs_from_molecule,
        )

        heterograph = Heterograph()
        heterograph['bond']['idxs'] =\
            get_bond_idxs_from_molecule(molecule)
        heterograph['angle']['idxs'] =\
            get_angle_idxs_from_molecule(molecule)
        heterograph['proper']['idxs'] =\
            get_proper_torsion_idxs_from_molecule(molecule)
        heterograph['improper']['idxs'] =\
            get_improper_torsion_idxs_from_molecule(molecule)
        heterograph['nonbonded']['idxs'] =\
            get_nonbonded_idxs_from_molecule(molecule)
        heterograph['onefour']['idxs'] =\
            get_onefour_idxs_from_molecule(molecule)

        return heterograph

    @classmethod
    def from_openff_molecule(cls, molecule: Molecule) -> NamedTuple:
        homograph = cls.homograph_from_openff_molecule(molecule)
        heterograph = cls.heterograph_from_openff_molecule(molecule)
        return cls(
            homograph=homograph, heterograph=heterograph,
        )

    @classmethod
    def from_smiles(cls, smiles: str) -> NamedTuple:
        molecule = Molecule.from_smiles(smiles)
        return cls.from_openff_molecule(molecule)
