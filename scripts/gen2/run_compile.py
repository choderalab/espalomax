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
    def __init__(self, path, partition="train", seed=2666):
        self.path = path
        self.partition = partition
        self._prepare()
  
    def _prepare(self):
        import os
        import pickle
        # base_path = "../data/qca_optimization/data/"
        base_path = self.path
        paths = os.listdir(base_path)
        paths = [base_path + path for path in paths]
        data = []
        for path in paths:
            _data = pickle.load(open(path, "rb"))
            data.append(_data)
        self.data = data
        idxs = list(range(len(self.data)))
        import random
        random.shuffle(idxs)
        n_te = int(0.1 * len(idxs))
        if self.partition == "train":
            self.idxs = idxs[:int(0.8 * n_te)]
        elif self.partition == "valid":
            self.idxs = idxs[int(0.8 * n_te) : int(0.9 * n_te)]
        elif self.partition == "test":
            self.idxs = idxs[int(0.9 * n_te):]

   
    def __iter__(self):
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
    dataloader = DataLoader(path="../data/qca_optimization/data/")

    g, _, __ = next(iter(dataloader))
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

    @jax.jit
    def step(state, g, x, u):
        nn_params = state.params
        grads = jax.grad(get_loss)(nn_params, g, x, u)
        state = state.apply_gradients(grads=grads)
        return state

    import optax
    optimizer = optax.adam(1e-3)
    optimizer = optax.apply_if_finite(optimizer, 5)

    nn_params = model.init(jax.random.PRNGKey(2666), g)
    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
         apply_fn=model.apply, params=nn_params, tx=optimizer,
    )


    import time
    time0 = time.time()
    compiled = []

    from concurrent.futures import ThreadPoolExecutor
    with futures.ThreadPoolExecutor() as pool:
        for g, x, u in dataloader.data:
            lowered = step.lower(state, g, x, u)
            compiled.append(pool.submit(lowered.compile))
    compiled = [fn.result() for fn in compiled]
    
    time1 = time.time()

    print(time1 - time0, flush=True)

    import tqdm
    import random
    for idx_batch in tqdm.tqdm(range(50000)):
        idxs = list(range(len(dataloader.data)))
        random.shuffle(idxs)
        for idx in idxs:
            g, x, u = dataloader.data[idx]
            state = compiled[idx](state, g, x, u)
            assert state.opt_state.notfinite_count <= 10
        save_checkpoint("_checkpoint", state, idx_batch)

if __name__ == "__main__":
    run()
