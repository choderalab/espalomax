from typing import Callable, Union, Dict
from dataclasses import field
import jax
import jax.numpy as jnp
from flax import linen as nn
from .graph import Graph, Heterograph
from jraph import GAT

IMPROPER_PERMUTATIONS = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
JANOSSY_POOLING_PARAMETERS = {
    "bonds": {"coefficients": 2},
    "angle": {"coefficients": 2},
    "proper": {"k": 6},
    "improper": {"k": 6},
}

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
            layer = getattr(self, "d_%s" % out_feature)
            if out_feature != "improper": # mirror symmetry
                h = layer(h.reshape(*h.shape[:-2], -1))\
                    + layer(jnp.flip(h, -2).reshape(*h.shape[:-2], -1))
            else:
                if len(heterograph['improper']['idxs'] > 0):
                    hs = [
                        layer(
                            h[..., jnp.array(permutation), :]
                            .reshape(*h.shape[:-2], -1)
                        )
                        for permutation in IMPROPER_PERMUTATIONS
                    ]

                    h = sum(hs)
                else:
                    h = jnp.array([], dtype=jnp.float32)
            for parameter in self.out_features[out_feature]:
                layer = getattr(self, "d_%s_%s" % (out_feature, parameter))
                parameters[out_feature][parameter] = layer(h)
        return parameters

class Parametrization(nn.Module):
    representation: Callable
    janossy_pooling: Callable

    def __call__(self, graph):
        homograph, heterograph = graph.homograph, graph.heterograph
        homograph = self.representation(homograph)
        parameters = self.janossy_pooling(heterograph, homograph.nodes)
        return parameters
