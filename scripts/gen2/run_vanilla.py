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


class DataLoader(object):
    def __init__(self):
        self._prepare()
    
    def _prepare(self):
        import os
        import pickle
        base_path = "../data/qca_optimization/data/"
        paths = os.listdir(base_path)
        paths = [base_path + path for path in paths]
        data = []
        for path in paths:
            _data = pickle.load(open(path, "rb"))
            data.append(_data)
        self.data = data
    
    def __iter__(self):
        self.idxs = list(range(len(self.data)))
        random.shuffle(self.idxs)
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if len(self.idxs) == 0:
            raise StopIteration

        idx = self.idxs.pop()
        g, x, u = self.data[idx]
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

    @jax.jit
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

    import tqdm
    for idx_batch in tqdm.tqdm(range(100)):
        for g, x, u in dataloader:
            state = step(state, g, x, u)

if __name__ == "__main__":
    run()
