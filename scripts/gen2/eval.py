import random
from openff.toolkit.topology import Molecule
import h5py
import numpy as onp
import jax
import jax.numpy as jnp
import jax_md
import espalomax as esp
from concurrent import futures

import warnings
warnings.filterwarnings("ignore")

BOHR_TO_NM = 0.0529177
HARTREE_TO_KCAL_PER_MOL = 627.5

def run():
    from run_compile import DataLoader
    dataloader = DataLoader(path="../data/qca_optimization/data/", partition="valid")
    
    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(64, 3),
        janossy_pooling=esp.nn.JanossyPooling(64, 3),
    )

    def get_loss(nn_params, g, x, u):
        ff_params = model.apply(nn_params, g)
        u_hat = esp.mm.get_energy(ff_params, x)
        u_hat = u_hat - u_hat.mean(0, keepdims=True)
        u = u - u.mean(0, keepdims=True)
        return jnp.abs(u - u_hat).mean()

    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("_checkpoint", None)
    params = state["params"]
    
    for g, x, u in dataloader:
        loss = get_loss(params, g, x, u) * HARTREE_TO_KCAL_PER_MOL
        print(loss, flush=True)

if __name__ == "__main__":
    run()
