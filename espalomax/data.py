from typing import List, Tuple
import numpy as onp
import jax
import jax.numpy as jnp
from .graph import Graph, dummy, batch

class PadToConstantDataLoader:
    data : Optional[List[Tuple[Graph, onp.ndarray, onp.ndarray]]] = None
    max_n_atoms: int = 512
    max_n_bonds: int = 512
    max_n_angles: int = 512
    max_n_propers: int = 512
    max_n_impropers: int = 512

    def append(self, _data):
        assert len(_data) == 3
        assert isinstance(_data[0], Graph)
        assert isinstance(_data[1], onp.ndarray)
        assert isinstance(_data[2], onp.ndarray)
        self.data.append(_data)

    def load(self, path):
        import pickle
        self.data = pickle.load(open(path, "rb"))
        return self

    def save(self, path):
        import pickle
        pickle.dump(open(path, "wb"))

    def __iter__(self):
        self.idxs = onp.random.randint(len(data))

    def __next__(self):
        sum_n_atoms = 0
        sum_n_bonds = 0
        sum_n_angles = 0
        sum_n_propers = 0
        sum_n_impropers = 0
        cache = []

        while all(
            sum_n_atoms <= self.max_n_atoms,
            sum_n_bonds <= self.max_n_bonds,
            sum_n_angles <= self.max_n_angles,
            sum_n_propers <= self.max_n_propers,
            sum_n_impropers <= self.max_n_impropers,
        ):
            if len(self.idxs) == 0:
                raise StopIteration
            else:
                idx = self.idxs.pop()
                g, x, u = self.data[idx]
                sum_n_atoms += g.n_atoms
                sum_n_bonds += g.n_bonds
                sum_n_angles += g.n_angles
                sum_n_propers += g.n_propers
                sum_n_impropers += g.impropers
                cache.append((g, x, u))

        dummy_n_atoms = self.max_n_atoms - sum_n_atoms
        dummy_n_bonds = self.max_n_bonds - sum_n_bonds
        dummy_n_angles = self.max_n_angles - sum_n_angles
        dummy_n_propers = self.max_n_propers - sum_n_propers
        dummy_n_impropers = self.max_n_impropers - sum_n_impropers

        gs, xs, us = zip(*cache)
        g_dummy = dummy(
            n_atoms=dummy_n_atoms,
            n_bonds=dummy_n_bonds,
            n_angles=dummy_n_angles,
            n_propers=dummy_n_propers,
            n_impropers=dummy_n_impropers,
        )
        x_dummy = onp.zeros(1, dummy_n_atoms, 3)
        u_dummy = onp.zeros(1, 3)

        gs = gs + (g_dummy,)
        xs = xs + (x_dummy,)
        us = us + (u_dummy,)

        g = batch(gs)
        x = jnp.concatenate(xs, 1)
        u = jnp.concatenate(us, 0)

        return
