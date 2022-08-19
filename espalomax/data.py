from typing import Optional, List, Tuple
import numpy as onp
import jax
import jax.numpy as jnp
from .graph import Graph, dummy, batch, heteromask

class PadToConstantDataLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self._prepare()

    def _prepare(self):
        max_n_atoms = max_n_bonds = max_n_angles \
            = max_n_propers = max_n_impropers = 0
        for g, _, __ in self.data:
            self.max_n_atoms = max(max_n_atoms, g.n_atoms) * self.batch_size
            self.max_n_bonds = max(max_n_bonds, g.n_bonds) * self.batch_size
            self.max_n_angles = max(max_n_angles, g.n_angles) * self.batch_size
            self.max_n_propers = max(max_n_propers, g.n_propers) * self.batch_size
            self.max_n_impropers = max(max_n_impropers, g.n_impropers) * self.batch_size

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
        import random
        idxs = list(range(len(self.data)))
        random.shuffle(idxs)
        self.idxs = idxs
        return self

    def __next__(self):
        if len(self.idxs) < self.batch_size:
            raise StopIteration

        idxs = self.idxs[:self.batch_size]
        self.idxs = self.idxs[self.batch_size:]

        cache = [self.data[idx] for idx in idxs]
        gs, xs, us = zip(*cache)
        sum_n_atoms = sum(g.n_atoms for g in gs)
        sum_n_bonds = sum(g.n_bonds for g in gs)
        sum_n_angles = sum(g.n_angles for g in gs)
        sum_n_propers = sum(g.n_propers for g in gs)
        sum_n_impropers = sum(g.n_impropers for g in gs)

        dummy_n_atoms = self.max_n_atoms - sum_n_atoms
        dummy_n_bonds = self.max_n_bonds - sum_n_bonds
        dummy_n_angles = self.max_n_angles - sum_n_angles
        dummy_n_propers = self.max_n_propers - sum_n_propers
        dummy_n_impropers = self.max_n_impropers - sum_n_impropers

        g_dummy = dummy(
            n_atoms=dummy_n_atoms,
            n_bonds=dummy_n_bonds,
            n_angles=dummy_n_angles,
            n_propers=dummy_n_propers,
            n_impropers=dummy_n_impropers,
            n_nonbonded=1,
            n_onefour=1,
        )

        x_dummy = onp.zeros((1, dummy_n_atoms, 3))
        u_dummy = onp.zeros((1, ))

        gs = gs + (g_dummy,)
        xs = xs + (x_dummy,)
        us = us + (u_dummy,)

        g = batch(gs)
        x = jnp.concatenate(xs, 1)
        u = jnp.stack(us, 0)
        m = heteromask(g)

        return g, x, u, m
