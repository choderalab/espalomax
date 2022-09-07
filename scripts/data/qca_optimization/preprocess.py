import random
from openff.toolkit.topology import Molecule
import h5py
import numpy as onp
import jax
import jax.numpy as jnp
import jax_md
import espalomax as esp
from concurrent import futures
import pickle
import warnings
warnings.filterwarnings("ignore")

BOHR_TO_NM = 0.0529177
HARTREE_TO_KCAL_PER_MOL = 627.5

def run(idx):
    file_handle = h5py.File("gen2.hdf5", "r")
    keys = list(file_handle.keys())
    key = keys[idx]
    record = file_handle[key]
    molecule = Molecule.from_mapped_smiles(
                record["smiles"][0].decode('utf-8'),
                allow_undefined_stereo=True,
    )
    g = esp.Graph.from_openff_molecule(molecule)
    x = jnp.array(record["x"]) * BOHR_TO_NM
    u = jnp.array(record["u"]) # * HARTREE_TO_KCAL_PER_MOL
    u0 = esp.mm.get_nonbonded_energy(molecule, x) / HARTREE_TO_KCAL_PER_MOL
    u = u - u0
    u0 = u.min()
    u = u - u0
    mask = (u < 0.1)
    if len(mask) > 1:
        x, u = x[mask], u[mask]
        data = (g, x, u)
        pickle.dump(data, open(f"data/{idx}.pkl", "wb"))

if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]))
