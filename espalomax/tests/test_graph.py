import pytest

def test_init_graph():
    import espalomax as esp
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("C")
    g = esp.Graph.from_openff_molecule(molecule)
