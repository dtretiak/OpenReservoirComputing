"""
Abstract base class for trainable readout layers.
"""

from abc import ABC, abstractmethod
import equinox as eqx
import jax
from jaxtyping import Array

class ReadoutBase(eqx.Module, ABC):
    """
    Base class dictating API for all implemented readout layers.

    Methods:
        readout(res_state)
        batch_readout(res_state)
    """

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