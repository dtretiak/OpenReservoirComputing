"""Implementations of common readout forms. """

from orc.readouts.base import ReadoutBase
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jaxtyping import Array, PRNGKeyArray, Float

jax.config.update("jax_enable_x64", True)

class LinearReadout(ReadoutBase):
    """Linear readout layer.

    Attributes:
        out_dim (int): reservoir output dimension
        res_dim (int): reservoir dimension
        w_or (Array): output matrix

    Methods:
        readout(res_state) -> output state
    """
    out_dim: int
    res_dim: int
    w_or: Array
    dtype: Float


    def __init__(self,
                 out_dim: int,
                 res_dim: int,
                 dtype: Float = jnp.float64,
                 *,
                 key: PRNGKeyArray,
                 ) -> None:
        """Initialize readout layer to zeros.
        
        Args:
            out_dim (int): reservoir output dimension
            res_dim (int): reservoir dimension
        """
        self.out_dim = out_dim
        self.res_dim = res_dim
        self.w_or = jnp.zeros((out_dim, res_dim), dtype=dtype)
        self.dtype = dtype

    def readout(self,
                res_state: Float[Array, "{self.res_dim}"]
                ) -> Float[Array, "{self.out_dim}"]:
        """Readout from reservoir state.
        
        Args:
            res_state (Array): reservoir state (shape=(res_dim,))

        Returns:
            (Array): output from reservoir (shape=(out_dim,))
        """
        return self.w_or @ res_state