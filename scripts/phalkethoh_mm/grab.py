import torch
import numpy as np
import espaloma as esp
import espalomax as espx
import pickle

BOHR_TO_NM = 0.0529177
HARTREE_TO_KCAL_PER_MOL = 627.5

def run(idx):
    '''
    lines = open("AlkEthOH_chain.smi").readlines()\
            + open("AlkEthOH_rings.smi").readlines()\
            + open("PhEthOH.smi").readlines()

    lines = [line.split(" ")[0] for line in lines]
    print(len(lines))
    line = lines[idx]
    '''
    line = "C"

    g = esp.Graph(line)
    gx = espx.Graph.from_openff_molecule(g.mol)

    esp.graphs.legacy_force_field.LegacyForceField("openff-1.2.0").parametrize(g)
    esp.data.md.MoleculeVacuumSimulation(
        forcefield="openff-1.2.0",
        n_samples=100,
        n_steps_per_sample=100000,
    ).run(g)
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph, terms=["n2", "n3"], suffix="_ref")
    x = g.heterograph.nodes['n1'].data['xyz'].detach().numpy().swapaxes(0, 1)
    u = g.heterograph.nodes['g'].data['u_ref'].detach().numpy().swapaxes(0, 1)

    pickle.dump((gx, x, u), open(f"data/{idx}.pkl", "wb"))



if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]))
