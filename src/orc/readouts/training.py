"""Training functions for different readouts.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from orc.readouts.layers import LinearReadout

jax.config.update("jax_enable_x64", True)

@eqx.filter_jit
def train_linear(read: LinearReadout,
          res_seq: Float[Array, "... {model.res_dim}"],
          targets: Float[Array, "... {model.out_dim}"],
          spinup: int,
          beta: float = 8.493901e-8) -> None:
    """Train a LinearReadout layer.

    Args: 
    res_seq (Array): reservoir sequence (shape=(seq_len, res_dim,))
    targets (Array): target data (shape=(seq_len, out_dim,))
    spinup (int): initial transient to discard
    beta (float): Tikhonov regularization parameter

    Returns:
    read (LinearReadout): trained LinearReadout model
    """
    lhs = (res_seq[spinup:].T @ res_seq[spinup:] 
            + beta * jnp.eye(read.res_dim, dtype=read.dtype))
    rhs = res_seq[spinup:].T @ targets[spinup:]
    cmat = jnp.linalg.lstsq(lhs, rhs, rcond=None)[0].T
    where = lambda m: m.w_or
    read = eqx.tree_at(where, read, cmat)
    return read