from openff.toolkit.topology import Molecule
import jax
import jax.numpy as jnp
import jax_md
import espalomax as esp


def run():
    molecule = Molecule.from_smiles("C")
    base_parameters = esp.graph.parameters_from_molecule(molecule)
    graph = esp.Graph.from_openff_molecule(molecule)

    model = esp.nn.Parametrization(
        representation=esp.nn.GraphAttentionNetwork(8, 3),
        janossy_pooling=esp.nn.JanossyPooling(8, 3),
    )
    nn_params = model.init(jax.random.PRNGKey(2666), graph)
    ff_params = model.apply(nn_params, graph)
    ff_params = esp.nn.to_jaxmd_mm_energy_fn_parameters(ff_params, base_parameters)

    from jax_md import space
    displacement_fn, shift_fn = space.periodic(25.0)

    from jax_md.mm import mm_energy_fn
    energy_fn, neighbor_fn = mm_energy_fn(displacement_fn, ff_params)
    

if __name__ == "__main__":
    run()
