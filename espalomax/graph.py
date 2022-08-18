from typing import NamedTuple, DefaultDict
from openff.toolkit.topology import Molecule
from jraph import GraphsTuple
import jax
import jax.numpy as jnp
from collections import defaultdict
from functools import partial
import jraph

def _lambda_none():
    return None

def _default_fn():
    return defaultdict(_lambda_none)

Heterograph = partial(defaultdict, _default_fn)


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
        nodes = jax.nn.one_hot(nodes, 118)  # TODO: use more features

        # get bonds
        senders = []
        receivers = []
        for bond in molecule.bonds:
            senders.append(bond.atom1_index)
            receivers.append(bond.atom2_index)

            # ensure homograph symmetry
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
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
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

        Examples
        --------
        >>> molecule = Molecule.from_smiles("C")
        >>> graph = Graph.from_openff_molecule(molecule)
        >>> graph.homograph.n_node
        5
        >>> graph.homograph.n_edge
        8

        >>> len(graph.heterograph["bond"]["idxs"])
        4
        >>> len(graph.heterograph["angle"]["idxs"])
        6

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
        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        return cls.from_openff_molecule(molecule)

    @property
    def n_atoms(self):
        """Number of atoms."""
        return int(sum(self.homograph.n_node))

    @property
    def n_bonds(self):
        """Number of bonds."""
        return len(self.heterograph["bond"]["idxs"])

    @property
    def n_angles(self):
        """Number of angles."""
        return len(self.heterograph["angle"]["idxs"])

    @property
    def n_propers(self):
        """Number of propers."""
        return len(self.heterograph["proper"]["idxs"])

    @property
    def n_impropers(self):
        """Number of impropers."""
        return len(self.heterograph["improper"]["idxs"])

def parameters_from_molecule(
        molecule: Molecule,
        base_forcefield: str = "openff_unconstrained-2.0.0.offxml",
) -> NamedTuple:
    """Get jax_md.mm.MMEnergyFnParameters from single molecule.

    Parameters
    ----------
    molecule : Molecule
        Input OpenFF molecule.
    base_forcefield : str
        Base force field for nonbonded terms and exceptions.

    Returns
    -------
    jax_md.mm.MMEnergyFnParameters
        Resulting parameters.
    """
    from openff.toolkit.typing.engines.smirnoff import ForceField
    molecule.assign_partial_charges("mmff94")
    forcefield = ForceField(base_forcefield)
    system = forcefield.create_openmm_system(
        molecule.to_topology(),
        charge_from_molecules=[molecule],
    )

    from jax_md.mm_utils import parameters_from_openmm_system
    parameters = parameters_from_openmm_system(system)
    return parameters

def batch(graphs):
    homographs = jraph.batch([graph.homograph for graph in graphs])
    offsets = homographs.n_node
    offsets = jnp.concatenate([jnp.array([0]), offsets[:-1]])
    offsets = jnp.cumsum(offsets)
    heterographs = Heterograph()
    for term in ["bond", "angle", "proper", "improper", "onefour", "nonbonded"]:
        heterographs[term]["idxs"] = jnp.concatenate(
            [
                graph.heterograph[term]["idxs"] + offsets[idx]
                for idx, graph in enumerate(graphs)
            ],
            axis=0
        )
    return Graph(homograph=homographs, heterograph=heterographs)

def dummy(
    n_atoms: int,
    n_bonds: int,
    n_angles: int,
    n_propers: int,
    n_impropers: int,
    n_nonbonded: int=0,
    n_onefour: int=0,
):
    """Return a dummy graph with specified number of entities.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    n_bonds : int
        Number of bonds.
    n_angles : int
        Number of angles.
    n_propers : int
        Number of proper torsions.
    n_impropers : int
        Number of imporper torsions.
    n_nonbonded : int(default=0)
        Number of nonbonded interations.
    n_onefour : int(default=0)
        Number of onefour interactions.

    Returns
    -------
    Graph
        Dummy graph.
    """
    nodes = jnp.zeros((n_atoms, 118), jnp.int32)
    senders = receivers = jnp.zeros(n_bonds, jnp.int32)

    homograph = GraphsTuple(
        nodes=nodes,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([n_bonds], jnp.int32),
        n_edge=jnp.array([n_bonds], jnp.int32),
        edges=None,
        globals=None,
    )

    heterograph = Heterograph()
    heterograph["bond"]["idxs"] = jnp.zeros((n_bonds, 2), jnp.int32)
    heterograph["angle"]["idxs"] = jnp.zeros((n_angles, 3), jnp.int32)
    heterograph["proper"]["idxs"] = jnp.zeros((n_propers, 4), jnp.int32)
    heterograph["improper"]["idxs"] = jnp.zeros((n_impropers, 4), jnp.int32)
    heterograph["nonbonded"]["idxs"] = jnp.zeros((n_nonbonded, 2), jnp.int32)
    heterograph["onefour"]["idxs"] = jnp.zeros((n_onefour, 2), jnp.int32)

    return Graph(homograph=homograph, heterograph=heterograph)

def heteromask(graph: Graph):
    mask = Heterograph()
    upper = jnp.cumsum(graph.homograph.n_node)
    lower = jnp.concatenate([jnp.array([0]), upper[:-1]])
    for term in ["bond", "angle", "proper", "improper", "nonbonded", "onefour"]:
        idxs = graph.heterograph[term]["idxs"]
        idxs = jnp.expand_dims(idxs, -1)
        _mask = (lower <= idxs) * (upper > idxs)
        _mask = _mask.prod(-2).argmax(-1)
        mask[term]["mask"] = _mask
    return mask
