import jax
import jax.numpy as jnp
from .graph import Heterograph

import math
BOND_PHASES = (1.5, 6.0)
ANGLE_PHASES = (0.0, math.pi)

class Geometry(object):
    @staticmethod
    def get_geometry_distance(heterograph, coordinates):
        x0 = coordinates[heterograph['bond']['idxs'][..., 0]]
        x1 = coordinates[heterograph['bond']['idxs'][..., 1]]
        length = ((x0 - x1) ** 2).sum(axis=-1) ** 0.5
        return length

    @staticmethod
    def get_geometry_bond(*args, **kwargs):
        return Geometry.get_geometry_distance(*args, **kwargs)

    @staticmethod
    def get_geometry_nonbonded(*args, **kwargs):
        return Geometry.get_geometry_distance(*args, **kwargs)

    @staticmethod
    def get_geometry_onefour(*args, **kwargs):
        return Geometry.get_geometry_distance(*args, **kwargs)

    @staticmethod
    def get_geometry_angle(heterograph, coordinates):
        x0 = coordinates[heterograph['angle']['idxs'][..., 0]]
        x1 = coordinates[heterograph['angle']['idxs'][..., 1]]
        x2 = coordinates[heterograph['angle']['idxs'][..., 2]]
        left = x1 - x0
        right = x1 - x2
        angle = jnp.arctan2(
            jnp.linalg.norm(jnp.cross(left, right), ord=2, axis=-1),
            jnp.sum(left * right, axis=-1),
        )
        return angle

    @staticmethod
    def get_geometry_torsion(heterograph, coordinates, torsion_type="proper"):
        x0 = coordinates[heterograph[torsion_type]['idxs'][..., 0]]
        x1 = coordinates[heterograph[torsion_type]['idxs'][..., 1]]
        x2 = coordinates[heterograph[torsion_type]['idxs'][..., 2]]
        x3 = coordinates[heterograph[torsion_type]['idxs'][..., 3]]

        r01 = x1 - x0
        r21 = x1 - x2
        r23 = x3 - x2

        n1 = jnp.cross(r01, r21)
        n2 = jnp.cross(r21, r23)

        rkj_normed = r21 / jnp.linalg.norm(r21, ord=2, axis=-1, keepdims=True)

        y = jnp.sum(jnp.multiply(jnp.cross(n1, n2), rkj_normed), axis=-1)
        x = jnp.sum(jnp.multiply(n1, n2), axis=-1)

        # choose quadrant correctly
        theta = jnp.arctan2(y, x)
        return theta

    @staticmethod
    def get_geometry_proper(heterograph, coordinates):
        return Geometry.get_geometry_torsion(
            heterograph, coordinates, torsion_type="proper"
        )

    @staticmethod
    def get_geometry_improper(heterograph, coordinates):
        return Geometry.get_geometry_torsion(
            heterograph, coordinates, torsion_type="improper"
        )

    @staticmethod
    def get_geometry(heterograph, coordinates):
        geometry = Heterograph()
        for term in heterograph.keys():
            geometry[term]['x'] = getattr(Geometry, "get_geometry_%s" % term)(
                heterograph, coordinates,
            )
        return geometry

def get_geometry(heterograph, coordinates):
    return Geometry.get_geometry(heterograph, coordinates)

class Energy(object):
    @staticmethod
    def get_energy_linear_mixture(x, coefficients, phases):
        # partition the dimensions
        # (, )
        b1 = phases[0]
        b2 = phases[1]

        # (batch_size, 1)
        k1 = jnp.exp(coefficients[:, 0][:, None])
        k2 = jnp.exp(coefficients[:, 1][:, None])

        # get the original parameters
        # (batch_size, )
        # k, b = linear_mixture_to_original(k1, k2, b1, b2)

        # (batch_size, 1)
        u1 = k1 * (x - b1) ** 2
        u2 = k2 * (x - b2) ** 2

        u = 0.5 * (u1 + u2)  # - k1 * b1 ** 2 - k2 ** b2 ** 2 + b ** 2

        return u

    @staticmethod
    def get_energy_bond(x, coefficients):
        return Energy.get_energy_linear_mixture(x, coefficients, BOND_PHASES)

    @staticmethod
    def get_energy_angle(x, coefficients):
        return Energy.get_energy_linear_mixture(x, coefficients, ANGLE_PHASES)

    @staticmethod
    def get_energy_torsion(
        x, k,
        periodicity=jnp.arange(1, 7), phases=jnp.zeros(6),
    ):
        n_theta = jnp.expand_dims(x, -1) * periodicity
        n_theta_minus_phases = n_theta - phases
        cos_n_theta_minus_phases = jnp.cos(n_theta_minus_phases)
        k = jnp.expand_dims(k, -2)
        energy = (
            jax.nn.relu(k) * (cos_n_theta_minus_phases + 1.0)
            - jax.nn.relu(0.0 - k) * (cos_n_theta_minus_phases - 1.0)
        ).sum(axis=-1)
        return energy

    @staticmethod
    def get_energy_improper(*args, **kwargs):
        return Energy.get_energy_torsion(*args, **kwargs)

    @staticmethod
    def get_energy_proper(*args, **kwargs):
        return Energy.get_energy_torsion(*args, **kwargs)

    @staticmethod
    def get_energy(parameters, geometry):
        energy = Heterograph()
        energy['bond']['u'] = Energy.get_energy_bond(
            geometry['bond']['x'], parameters['bond']['coefficients'],
        )

        energy['angle']['u'] = Energy.get_energy_angle(
            geometry['angle']['x'], parameters['angle']['coefficients'],
        )

        energy['proper']['u'] = Energy.get_energy_proper(
            geometry['proper']['x'], parameters['proper']['k'],
        )

        energy['improper']['u'] = Energy.get_energy_improper(
            geometry['improper']['x'], parameters['improper']['k'],
        )

        return energy

def get_energy(parameters, geometry):
    return Energy.get_energy(parameters, geometry)
