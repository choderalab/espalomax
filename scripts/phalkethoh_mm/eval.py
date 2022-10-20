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
    dataloader = DataLoader(path="data/", partition="all")
    
    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(128, 6),
        janossy_pooling=esp.nn.JanossyPooling(128, 2),
    )

    def get_loss(nn_params, g, x, u):
        ff_params = model.apply(nn_params, g)
        u_hat = esp.mm.get_energy(ff_params, x)
        u_hat = u_hat - u_hat.mean(0, keepdims=True)
        u = u - u.mean(0, keepdims=True)
        return jnp.abs(u - u_hat).mean()

    def save_traj(nn_params, g, x, u, idx):
        ff_params = model.apply(nn_params, g)
        u_hat = esp.mm.get_energy(ff_params, x)
        u_hat = u_hat - u_hat.mean(0, keepdims=True)
        u = u - u.mean(0, keepdims=True)
        u_hat = onp.array(u_hat)
        u = onp.array(u)
        # onp.save("traj/u%s.npy" % idx, u)
        # onp.save("traj/u_hat%s.npy" % idx, u_hat)



    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("__checkpoint", target=None)
    params = state["params"]
    print(params)

    for idx, (g, x, u) in enumerate(dataloader):
        # save_traj(params, g, x, u, idx)
        loss = get_loss(params, g, x, u) # * HARTREE_TO_KCAL_PER_MOL
        print(loss, flush=True)

if __name__ == "__main__":
    run()
