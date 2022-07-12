import pytest

def test_init_graph():
    import espalomax as esp
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("C")
    g = esp.Graph.from_openff_molecule(molecule)

def test_parameters_from_molecule():
    import espalomax as esp
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("C")
    parameters = esp.graph.parameters_from_molecule(molecule)
    from jax_md.mm import check_parameters
    check_parameters(parameters)
