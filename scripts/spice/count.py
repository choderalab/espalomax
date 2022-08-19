import numpy as np
import pickle
from run import DataLoader

def run():
    file_handle = open("data.pkl", "rb")
    data = pickle.load(file_handle)
    _n_atoms = []
    _n_bonds = []
    _n_angles = []
    _n_torsions = []

    for g, _, __ in data:
        n_atoms, n_bonds, n_angles, n_torsions = (
            int(g.homograph.n_node),
            len(g.heterograph["bond"]["idxs"]),
            len(g.heterograph["angle"]["idxs"]),
            len(g.heterograph["proper"]["idxs"]) + len(g.heterograph["improper"]["idxs"]),
        )

        _n_atoms.append(n_atoms)
        _n_bonds.append(n_bonds)
        _n_angles.append(n_angles)
        _n_torsions.append(n_torsions)

    print(max(_n_atoms), max(_n_bonds), max(_n_angles), max(_n_torsions))
    print(min(_n_atoms), min(_n_bonds), min(_n_angles), min(_n_torsions))

if __name__ == "__main__":
    run()
