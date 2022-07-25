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
        for key in list(self.file_handle.keys())[:10]:
            record = self.file_handle[key]
            molecule = Molecule.from_mapped_smiles(
                        self.file_handle[key]["smiles"][0].decode('UTF-8'),
                        allow_undefined_stereo=True,
            )
            g = esp.Graph.from_openff_molecule(molecule)
            x = jnp.array(record["conformations"]) * BOHR_TO_NM
            u = jnp.array(record["formation_energy"]) * HARTREE_TO_KCAL_PER_MOL
        data.append((g, x, u))
        data = tuple(data)
        self.data = data

    def __iter__(self):
        self.idxs = list(range(len(self.data)))
        random.shuffle(self.idxs)
        return self

    def __next__(self):
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

    base_parameters = esp.graph.parameters_from_molecule(molecule)
    nn_params = model.init(jax.random.PRNGKey(2666), g)
    ff_params = model.apply(nn_params, g)
    ff_params = esp.nn.to_jaxmd_mm_energy_fn_parameters(ff_params, base_parameters)

    from jax_md import space
    displacement_fn, shift_fn = space.free()

    from jax_md.mm import mm_energy_fn
    energy_fn, neighbor_fn = mm_energy_fn(
        displacement_fn, ff_params,
        space_shape=space.free,
        use_neighbor_list=False,
        box_size=None,
        use_multiplicative_isotropic_cutoff=False,
        use_dsf_coulomb=False,
        neighbor_kwargs={},
    )

    def u_from_nn_params(nn_params):
        ff_params = model.apply(nn_params, graph)
        ff_params = esp.nn.to_jaxmd_mm_energy_fn_parameters(ff_params, base_parameters)
        return energy_fn(
            jax.random.normal(key=jax.random.PRNGKey(2666), shape=(6, 3)),
            parameters=ff_params,
        )







if __name__ == "__main__":
    run()
