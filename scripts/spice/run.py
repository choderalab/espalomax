import random
from openff.toolkit.topology import Molecule
import h5py
import numpy as onp
import jax
import jax.numpy as jnp
import jax_md
import espalomax as esp

import warnings
warnings.filterwarnings("ignore")

BOHR_TO_NM = 0.0529177
HARTREE_TO_KCAL_PER_MOL = 627.5


class DataLoader(object):
    def __init__(self):
        self.file_handle = h5py.File("SPICE.hdf5", "r")
        self._prepare()

    def _prepare(self):
        data = []
        for key in list(self.file_handle.keys())[:1]:
            record = self.file_handle[key]
            molecule = Molecule.from_mapped_smiles(
                        self.file_handle[key]["smiles"][0].decode('UTF-8'),
                        allow_undefined_stereo=True,
            )
            g = esp.Graph.from_openff_molecule(molecule)
            x = jnp.array(record["conformations"]) * BOHR_TO_NM
            u = jnp.array(record["formation_energy"]) * HARTREE_TO_KCAL_PER_MOL
            u0 = esp.mm.get_nonbonded_energy(molecule, x)
            u = u - u0
        data.append((g, x, u))
        data = tuple(data)
        self.data = data

    def __iter__(self):
        self.idxs = list(range(len(self.data)))
        random.shuffle(self.idxs)
        return self

    def __next__(self):
        if len(self.idxs) == 0:
            raise StopIteration

        idx = self.idxs.pop()
        g, x, u = self.data[idx]
        if len(x) != 50:
            idxs = onp.randint(high=len(x), size=50)
            x, u = x[idxs], u[idxs]
        return g, x, u

def run():
    dataloader = DataLoader()
    g, _, __ = next(iter(dataloader))
    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(64, 3),
        janossy_pooling=esp.nn.JanossyPooling(64, 3),
    )

    def get_loss(nn_params, g, x, u):
        ff_params = model.apply(nn_params, g)
        u_hat = esp.mm.get_energy(ff_params, x)
        return jnp.abs(u - u_hat).mean()

    def step(state, g, x, u):
        nn_params = state.params
        grads = jax.grad(get_loss)(nn_params, g, x, u)
        state = state.apply_gradients(grads=grads)
        return state

    import optax
    optimizer = optax.adam(1e-3)

    nn_params = model.init(jax.random.PRNGKey(2666), g)
    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
         apply_fn=model.apply, params=nn_params, tx=optimizer,
    )

    for g, x, u in dataloader:
        state = step(state, g, x, u)

if __name__ == "__main__":
    run()
