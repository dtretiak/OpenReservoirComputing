from abc import ABC, abstractmethod
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Float

class ReadoutBase(eqx.Module, ABC):
    """
    Base class dictating API for all implemented readout layers.

    Methods:
    readout(res_state)
    batch_readout(res_state)
    """
    out_dim: int
    res_dim: int
    dtype: Float

    def __init__(self, out_dim, res_dim, dtype=jnp.float64):
        """Ensures in dim, res dim,  and dtype are correct type."""
        self.res_dim = res_dim
        self.out_dim = out_dim
        self.dtype = dtype
        if not isinstance(res_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        if not isinstance(out_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        self.dtype = dtype
        if not (dtype == jnp.float64 or dtype == jnp.float32):
            raise TypeError("dtype must be jnp.float64 of jnp.float32.")
        
    @abstractmethod
    def readout(self,
                res_state: Array,
                ) -> Array:
        """Readout from reservoir state.

        Args:
            res_state (Array): reservoir state (shape=(res_dim,))
        
        Returns:
        ----------
        (Array): output from reservoir state (shape=(out_dim,))
        """
        pass

    def batch_readout(self,
                      res_state: Array,
                      ) -> Array:
        """Batch apply readout from reservoir states.

        Args:
        res_state (Array): reservoir state (shape=(batch_dim, res_dim,))
        
        Returns:
        ----------
        (Array): output from reservoir states (shape=(batch_dim, out_dim,))
        """
        return eqx.filter_vmap(self.readout)(res_state)
    


class LinearReadout(ReadoutBase):
    """Linear readout layer.

    Attributes:
    out_dim (int): reservoir output dimension
    res_dim (int): reservoir dimension
    wout (Array): output matrix

    Methods:
    readout(res_state) -> output state
    """
    out_dim: int
    res_dim: int
    wout: Array
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
        super().__init__(out_dim=out_dim, res_dim=res_dim, dtype=dtype)
        self.out_dim = out_dim
        self.res_dim = res_dim
        self.wout = jnp.zeros((out_dim, res_dim), dtype=dtype)
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
        return self.wout @ res_state