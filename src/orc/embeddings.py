from abc import ABC, abstractmethod
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Float

class EmbedBase(eqx.Module, ABC):
    """
    Base class dictating API for all implemented embedding layers.

    Methods:
    embed(in_state)
    batch_embed(in_state)
    """
    in_dim: int
    res_dim: int
    dtype: Float

    def __init__(self, in_dim, res_dim, dtype=jnp.float64):
        """Ensures in dim, res dim,  and dtype are correct type."""
        self.res_dim = res_dim
        self.in_dim = in_dim
        self.dtype = dtype
        if not isinstance(res_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        if not isinstance(in_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        self.dtype = dtype
        if not (dtype == jnp.float64 or dtype == jnp.float32):
            raise TypeError("dtype must be jnp.float64 of jnp.float32.")

    @abstractmethod
    def embed(self,
                in_state: Array,
                ) -> Array:
        """Embed input signal to reservoir dimension.

        Args:
        in_state (Array): input state (shape=(in_dim,))
        
        Returns:
        ----------
        (Array): embedded input state to reservoir dimension(shape=(res_dim,))
        """
        pass

    def batch_embed(self,
                      in_state: Array,
                      ) -> Array:
        """Batch apply readout from reservoir states.

        Args:
        in_state (Array): input state (shape=(batch_dim, in_dim,))
        
        Returns:
        ----------
        (Array): embedded input states to reservoir
            (shape=(batch_dim, res_dim,))
        """
        return eqx.filter_vmap(self.embed)(in_state)
    
class LinearEmbedding(EmbedBase):
    """Linear embedding layer.
    
    Attributes:
    in_dim (int): reservoir output dimension
    res_dim (int): reservoir dimension
    scaling (float): min/max values of input matrix
    win (Array): input matrix

    Methods:
    embed(in_state) -> reservoir input
    """
    in_dim: int
    res_dim: int
    scaling: float
    win: Array
    dtype: Float

    def __init__(self,
                 in_dim: int,
                 res_dim: int,
                 scaling: float,
                 dtype: Float = jnp.float64,
                 *,
                 key: PRNGKeyArray,
                 ) -> None:
        
        super().__init__(in_dim=in_dim, res_dim=res_dim, dtype=dtype)
        self.in_dim = in_dim
        self.res_dim = res_dim
        self.scaling = scaling
        self.dtype = dtype
        self.win = jax.random.uniform(key,
                                      (res_dim, in_dim),
                                      minval=-scaling,
                                      maxval=scaling,
                                      dtype=dtype)
    
    def embed(self, in_state: Array) -> Array:
        """Embed into reservoir dimension.
        
        Args:
        in_state (Array): input state (shape=(in_dim,))

        Returns:
        (Array): input to reservoir (shape=(res_dim,))
        """
        return self.win @ in_state
