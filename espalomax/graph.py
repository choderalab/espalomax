from typing import NamedTuple, DefaultDict
from openff.toolkit.topology import Molecule
from jraph import GraphsTuple
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
# from typing import Dict, DefaultDict
from collections import defaultdict
from functools import partial

Heterograph = partial(defaultdict, lambda: defaultdict(lambda: None))

class Graph(NamedTuple):
    """An espaloma graph---a homograph that stores the node topology and
    a heterograph that stores higher-order node topology.

    Attributes
    ----------
    homograph : jraph.GraphsTuple
        Node topology.
    heterograph : Heterograph
        Higher-order node topology

    Methods
    -------
    homograph_from_openff_molecule(molecule)
        Construct a homograph from OpenFF Molecule.
    heterograph_from_openff_molecule(molecule)
        Construct a heterograph from OpenFF Molecule.
    from_openff_molecule(molecule)
        Construct a graph (with homo- and heterograph) from OpenFF Molecule.
    from_smiles(str)
        Construct a graph (with homo- and heterograph) from SMILES string.

    """
    homograph: GraphsTuple
    heterograph: Heterograph

    @staticmethod
    def homograph_from_openff_molecule(molecule: Molecule) -> GraphsTuple:
        """Create a homograph from OpenFF Molecule object.

        Parameters
        ----------
        molecule : Molecule
            Input molecule.

        Returns
        -------
        GraphsTuple
            The homograph.
        """
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
        """Construct heterograph from an OpenFF molecule.

        Parameters
        ----------
        molecule : Molecule
            Input OpenFF molecule.

        Returns
        -------
        Heterograph
            The constructed heterograph.
        """
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
        """Construct Graph from OpenFF Molecule.

        Parameters
        ----------
        molecule : Molecule
            Input molecule.

        Returns
        -------
        Graph
            Espaloma graph.
        """
        homograph = cls.homograph_from_openff_molecule(molecule)
        heterograph = cls.heterograph_from_openff_molecule(molecule)
        return cls(
            homograph=homograph, heterograph=heterograph,
        )

    @classmethod
    def from_smiles(cls, smiles: str) -> NamedTuple:
        """Construct Graph from SMILES string.

        Parameters
        ----------
        smiles : str
            Input SMILES string.

        Returns
        -------
        Graph
            Resulting molecule.
        """
        molecule = Molecule.from_smiles(smiles)
        return cls.from_openff_molecule(molecule)