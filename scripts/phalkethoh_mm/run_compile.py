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
        random.seed(2666)
        random.shuffle(idxs)
        n_te = int(0.1 * len(idxs))
        if self.partition == "train":
            self.idxs = idxs[:int(8 * n_te)]
        elif self.partition == "valid":
            self.idxs = idxs[int(8 * n_te) : int(9 * n_te)]
        elif self.partition == "test":
            self.idxs = idxs[int(9 * n_te):]
        elif self.partition == "all":
            self.idxs = idxs

   
    def __iter__(self):
        random.shuffle(self.idxs)
        return self

    def __len__(self):
        return len(self.idxs)

    def __next__(self):
        if len(self.idxs) == 0:
            raise StopIteration

        idx = self.idxs.pop()
        g, x, u = self.data[idx]
        return g, x, u

def run():
    dataloader = DataLoader(path="data/", partition="all")

    g, _, __ = next(iter(dataloader))
    model = esp.nn.Parametrization(
        # representation=esp.nn.GraphSageModel(128, 6),
        representation=esp.nn.GraphAttentionNetwork(128, 6),
        janossy_pooling=esp.nn.JanossyPooling(128, 2),
    )

    def get_loss(nn_params, g, x, u):
        ff_params = model.apply(nn_params, g)
        u_hat = esp.mm.get_energy(ff_params, x)
        u_hat = u_hat - u_hat.mean(0, keepdims=True)
        u = u - u.mean(0, keepdims=True)
        return ((u - u_hat) ** 2).mean()

    @jax.jit
    def step(state, g, x, u):
        nn_params = state.params
        grads = jax.grad(get_loss)(nn_params, g, x, u)
        state = state.apply_gradients(grads=grads)
        return state

    import optax
    optimizer = optax.chain(
        optax.additive_weight_decay(1e-5),
        optax.clip(1.0),
        optax.adam(learning_rate=1e-4),
    )

    nn_params = model.init(jax.random.PRNGKey(2666), g)
    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
         apply_fn=model.apply, params=nn_params, tx=optimizer,
    )

    '''
    import time
    time0 = time.time()
    compiled = {}

    from concurrent.futures import ThreadPoolExecutor
    idxs = tuple(dataloader.idxs)
    compiled = {}
    with futures.ThreadPoolExecutor() as pool:
        for idx in idxs:
            g, x, u = dataloader.data[idx]
            print(x, u)
            key = (g.n_atoms, g.n_bonds, g.n_angles, g.n_propers, g.n_impropers, x.shape[0])
            if key not in compiled:
                lowered = step.lower(state, g, x, u)
                compiled[key] = pool.submit(lowered.compile)
    print(len(compiled), "fuck", flush=True)
    compiled = {key: fn.result() for key, fn in compiled.items()}
    
    time1 = time.time()

    print(time1 - time0, flush=True)
    '''

    idxs = tuple(dataloader.idxs)

    import tqdm
    import random
    for idx_batch in range(50000):
        _idxs = list(idxs)
        random.shuffle(_idxs)
        for idx in _idxs:
            g, x, u = dataloader.data[idx]
            # key = (g.n_atoms, g.n_bonds, g.n_angles, g.n_propers, g.n_impropers, x.shape[0])
            # state = compiled[key](state, g, x, u)
            state = step(state, g, x, u)
            print(get_loss(state.params, g, x, u))
        save_checkpoint("__checkpoint", state, idx_batch, keep_every_n_steps=10)

if __name__ == "__main__":
    run()
