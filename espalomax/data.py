from typing import Optional, List, Tuple
import numpy as onp
import jax
import jax.numpy as jnp
from .graph import Graph, dummy, batch

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
        sum_n_atoms = 0
        sum_n_bonds = 0
        sum_n_angles = 0
        sum_n_propers = 0
        sum_n_impropers = 0
        cache = []

        while True:
            if len(self.idxs) == 0:
                raise StopIteration
            else:
                idx = self.idxs[-1]
                g, x, u = self.data[idx]
                _sum_n_atoms = sum_n_atoms + g.n_atoms
                _sum_n_bonds = sum_n_bonds + g.n_bonds
                _sum_n_angles = sum_n_angles + g.n_angles
                _sum_n_propers = sum_n_propers + g.n_propers
                _sum_n_impropers = sum_n_impropers + g.n_impropers

                if all(
                    (
                        _sum_n_atoms <= self.max_n_atoms,
                        _sum_n_bonds <= self.max_n_bonds,
                        _sum_n_angles <= self.max_n_angles,
                        _sum_n_propers <= self.max_n_propers,
                        _sum_n_impropers <= self.max_n_impropers,
                    ),
                ):
                    sum_n_atoms = _sum_n_atoms
                    sum_n_bonds = _sum_n_bonds
                    sum_n_angles = _sum_n_angles
                    sum_n_propers = _sum_n_propers
                    sum_n_impropers = _sum_n_impropers
                    cache.append((g, x, u))
                    self.idxs.pop()

                else:
                    sum_n_atoms = 0
                    sum_n_bonds = 0
                    sum_n_angles = 0
                    sum_n_propers = 0
                    sum_n_impropers = 0
                    break


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
            n_nonbonded=1,
            n_onefour=1,
        )

        x_dummy = onp.zeros((1, dummy_n_atoms, 3))
        u_dummy = onp.zeros((1, 1))

        gs = gs + (g_dummy,)
        xs = xs + (x_dummy,)
        us = us + (u_dummy,)

        g = batch(gs)

        x = jnp.concatenate(xs, 1)
        u = jnp.concatenate(us, 0)

        return g, x, u
