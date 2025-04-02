"""
Abstract base class for reservoirs.
"""

from abc import ABC, abstractmethod
import equinox as eqx
import jax
from jaxtyping import Array


class ReservoirBase(eqx.Module, ABC):
    """
    Base class dictating API for all implemented reservoirs.

    Methods:
        advance(in_vars, res_state)
        batch_advance(in_vars, res_state)
    """

    @abstractmethod
    def advance(self,
                in_vars: Array,
                res_state: Array
                ) -> Array:
        """Advance the reservoir given inputs and current state.

        Args:
            in_vars (Array): inputs to reservoir (shape=(in_dim,))
            res_state (Array): reservoir state (shape=(res_dim,))
        
        Returns:
        ----------
        (Array): updated reservoir state (shape=(res_dim,))
        """
        pass

    def batch_advance(self,
                      in_vars: Array,
                      res_state: Array
                      ) -> Array:
        """Batch advance the reservoir given inputs and current state.

        Args:
            in_vars (Array): reservoir inputs
                (shape=(batch_size, in_dim,))
            res_state (Array): reservoir state
                (shape=(batch_size, res_dim,))
        
        Returns:
            (Array): updated reservoir state (shape=(batch_size, res_dim,))
        """
        return eqx.filter_vmap(self.advance)(in_vars, res_state)