from typing import Callable, Union, Dict
from dataclasses import field
import jax
import jax.numpy as jnp
from flax import linen as nn
from .graph import Graph, Heterograph
from jraph import GAT

IMPROPER_PERMUTATIONS = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
JANOSSY_POOLING_PARAMETERS = {
    "bond": {"coefficients": 2},
    "angle": {"coefficients": 2},
    "proper": {"k": 6},
    "improper": {"k": 6},
}

import math
BOND_PHASES = (0.00, 1.0)
ANGLE_PHASES = (0.0, math.pi)

class AttentionQueryFn(nn.Module):
    hidden_features: int

    @nn.compact
    def __call__(self, nodes):
        return jnp.expand_dims(nn.Dense(self.hidden_features)(nodes), -1)

class AttentionLogitFn(nn.Module):
    n_heads: int=4

    @nn.compact
    def __call__(self, sent_attributes, recived_attributes, edges=None):
        concatenated_attributes = jnp.concatenate(
            [sent_attributes, recived_attributes], axis=-1
        )

        return jax.nn.leaky_relu(
            nn.Dense(self.n_heads)(concatenated_attributes),
            negative_slope=0.2,
        )

class NodeUpdateFn(nn.Module):
    hidden_features: int
    last: False
    activation: Callable=jax.nn.elu

    @nn.compact
    def __call__(self, nodes):
        if self.last:
            return self.activation(nodes.mean(axis=-1))
        else:
            return self.activation(jnp.reshape(nodes, (*nodes.shape[:-2], -1)))


class _GAT(nn.Module):
    attention_query_fn: Callable
    attention_logit_fn: Callable
    node_update_fn: Callable

    def __call__(self, graph):
        return GAT(
            attention_query_fn=self.attention_query_fn,
            attention_logit_fn=self.attention_logit_fn,
            node_update_fn=self.node_update_fn,
        )(graph)

class GraphAttentionNetwork(nn.Module):
    hidden_features: int
    depth: int
    n_heads: int=4

    def setup(self):
        layers = []
        for idx_depth in range(self.depth):
            last = (idx_depth + 1 == self.depth)
            layers.append(
                _GAT(
                    attention_query_fn=AttentionQueryFn(self.hidden_features),
                    attention_logit_fn=AttentionLogitFn(self.n_heads),
                    node_update_fn=NodeUpdateFn(self.hidden_features, last=last),
                )
            )
        self.layers = nn.Sequential(layers)

    def __call__(self, graph):
        return self.layers(graph)

class JanossyPooling(nn.Module):
    hidden_features: int
    depth: int
    out_features: Union[Dict, None]=field(
        default_factory=lambda: JANOSSY_POOLING_PARAMETERS
    )
    activation: Callable = jax.nn.elu

    def setup(self):
        for out_feature in self.out_features.keys():
            layers = []
            for idx_depth in range(self.depth):
                layers.append(nn.Dense(self.hidden_features))
                layers.append(self.activation)
            layers = nn.Sequential(layers)
            setattr(self, "d_%s" % out_feature, layers)

            for parameter, dimension in self.out_features[out_feature].items():
                setattr(
                    self, "d_%s_%s" % (out_feature, parameter),
                    nn.Dense(dimension),
                )

    def __call__(self, heterograph: Heterograph, nodes: jnp.ndarray):
        parameters = Heterograph()
        for out_feature in self.out_features.keys():
            h = nodes[heterograph[out_feature]['idxs']]
            if jnp.size(h) > 0:
                layer = getattr(self, "d_%s" % out_feature)
                if out_feature != "improper": # mirror symmetry
                    h = layer(h.reshape(*h.shape[:-2], -1))\
                        + layer(jnp.flip(h, -2).reshape(*h.shape[:-2], -1))
                else:
                    hs = [
                        layer(
                            h[..., jnp.array(permutation), :]
                            .reshape(*h.shape[:-2], -1)
                        )
                        for permutation in IMPROPER_PERMUTATIONS
                    ]

                    h = sum(hs)
            else:
                h = jnp.array([[]], dtype=jnp.float32)

            for parameter in self.out_features[out_feature]:
                layer = getattr(self, "d_%s_%s" % (out_feature, parameter))

                if jnp.size(h) > 0:
                    parameters[out_feature][parameter] = layer(h)
                else:
                    parameters[out_feature][parameter] = jnp.array([[]], dtype=jnp.float32)

                parameters[out_feature]["idxs"] = heterograph[out_feature]["idxs"]
        return parameters

class Parametrization(nn.Module):
    representation: Callable
    janossy_pooling: Callable

    def __call__(self, graph):
        homograph, heterograph = graph.homograph, graph.heterograph
        homograph = self.representation(homograph)
        parameters = self.janossy_pooling(heterograph, homograph.nodes)
        return parameters

def linear_mixture_to_original(coefficients, phases):
    k1 = coefficients[..., 0]
    k2 = coefficients[..., 1]
    b1 = phases[0]
    b2 = phases[1]

    k = jnp.exp(k1) + jnp.exp(k2)
    b = (k1 * b1 + k2 * b2) / (k + 1e-7)
    return k, b

def to_jaxmd_mm_energy_fn_parameters(parameters, to_replace=None):
    from jax_md.mm import (
        MMEnergyFnParameters,
        HarmonicBondParameters,
        HarmonicAngleParameters,
        PeriodicTorsionParameters,
    )

    proper_idxs = parameters["proper"]["idxs"]
    proper_k = parameters["proper"]["k"]
    improper_idxs = parameters["improper"]["idxs"]
    improper_k = parameters["improper"]["k"]

    if jnp.size(proper_idxs) == 0:
        proper_idxs = jnp.zeros((0, 4), jnp.int32)
        proper_k = jnp.zeros((0, 6), jnp.float32)

    if jnp.size(improper_idxs) == 0:
        improper_idxs = jnp.zeros((0, 4), jnp.int32)
        improper_k = jnp.zeros((0, 6), jnp.float32)


    epsilon_bond, length_bond = linear_mixture_to_original(
            parameters['bond']['coefficients'], BOND_PHASES
    )

    harmonic_bond_parameters = HarmonicBondParameters(
        particles=parameters['bond']['idxs'],
        epsilon=epsilon_bond,
        length=length_bond,
    )

    epsilon_angle, length_angle = linear_mixture_to_original(
            parameters['angle']['coefficients'], ANGLE_PHASES,
    )

    harmonic_angle_parameters = HarmonicAngleParameters(
        particles=parameters['angle']['idxs'],
        epsilon=epsilon_angle,
        length=length_angle,
    )

    periodic_torsion_parameters = PeriodicTorsionParameters(
        particles=jnp.concatenate(
            [
                jnp.repeat(proper_idxs, 6, 0),
                jnp.repeat(improper_idxs, 6, 0),
            ],
        ),
        amplitude=jnp.concatenate(
            [
                proper_k.flatten(),
                improper_k.flatten(),
            ],
        ),
        periodicity=jnp.tile(
            jnp.arange(1, 7),
            len(parameters["proper"]["idxs"]) \
            + len(parameters["improper"]["idxs"])
        ),
        phase=jnp.zeros(
            6 * len(parameters["proper"]["idxs"]) \
            + 6 * len(parameters["improper"]["idxs"])
        ),
    )

    if to_replace is None:
        return MMEnergyFnParameters(
            harmonic_bond_parameters=harmonic_bond_parameters,
            harmonic_angle_parameters=harmonic_angle_parameters,
            periodic_torsion_parameters=periodic_torsion_parameters,
        )
    else:
        return to_replace._replace(
            harmonic_bond_parameters=harmonic_bond_parameters,
            harmonic_angle_parameters=harmonic_angle_parameters,
            periodic_torsion_parameters=periodic_torsion_parameters,
        )
