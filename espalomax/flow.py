import jax
import jax.numpy as jnp
from flax import linen as nn

JANOSSY_POOLING_PARAMETERS = {
    "bond": {"coefficients": 2},
    "angle": {"coefficients": 2},
    "proper": {"k": 6},
    "improper": {"k": 6},
}

from .nn import JANOSSY_POOLING_PARAMETERS
ORDER = 16

def get_polynomial_parameters(
    janossy_pooling_parameters: dict = JANOSSY_POOLING_PARAMETERS,
    order: int = ORDER,
) -> dict:
    return {
        key_out: {
            key_in: value_in * order for key_in, value_in in value_out.items()
        }
        for key_out, value_out in janossy_pooling_parameters.items()
    }

def constraint_polynomial_parameters(
    polynomial_parameters: dict,
):
    return {
        key_out: {
            key_in: jax.nn.log_softmax(value_in) if key_in == "coefficients" else 
                    jax.nn.tanh(value_in) if key_in == "k" else value_in
            for key_in, value_in in value_out.items()
        }
        for key_out, value_out in polynomial_parameters.items()
    }

def eval_polynomial(
    t: float,
    polynomial_parameters: dict,
    order: int = ORDER,
):
    return {
        key_out: {
            key_in: jnp.polyval(value_in.reshape(order, *value_in.shape[:-1], -1), t) if key_in != "idxs" else value_in
            for key_in, value_in in value_out.items()
        }
        for key_out, value_out in polynomial_parameters.items()
    }
