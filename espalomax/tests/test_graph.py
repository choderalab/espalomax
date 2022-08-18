import pytest

def test_init_graph():
    import espalomax as esp
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("C")
    g = esp.Graph.from_openff_molecule(molecule)

def test_parameters_from_molecule():
    import espalomax as esp
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("C=C")
    parameters = esp.graph.parameters_from_molecule(molecule)
    from jax_md.mm import check_parameters
    check_parameters(parameters)

def test_batching():
    import espalomax as esp
    from openff.toolkit.topology import Molecule
    g0 = esp.Graph.from_openff_molecule(Molecule.from_smiles("C=C"))
    g1 = esp.Graph.from_openff_molecule(Molecule.from_smiles("C=CC"))
    g = esp.graph.batch((g0, g1))

def test_heteromask():
    import espalomax as esp
    from openff.toolkit.topology import Molecule
    g0 = esp.Graph.from_openff_molecule(Molecule.from_smiles("C=C"))
    g1 = esp.Graph.from_openff_molecule(Molecule.from_smiles("C=CC"))
    g = esp.graph.batch((g0, g1))
    heteromask = esp.graph.heteromask(g)
