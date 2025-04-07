"""Basic ESN model for forecasting."""

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray, Float

from orc.readouts.layers import LinearReadout
from orc.readouts.training import train_linear
from orc.reservoirs.drivers import ESNReservoir

jax.config.update("jax_enable_x64", True)

class ESN(eqx.Module):
    """Basic implementation of ESN for forecasting.

    Attributes:
    res (ESNReservoir): esn driver
    read (LinearReadout): linear readout layer
    state_dim (int): input/output dimension of reservoir
    res_dim (int): reservoir dimension
    
    Methods:
    force(in_seq, res_state) -> sequence of reservoir states
    forecast(fcast_len, res_state) -> forecasted output states
    """
    res: ESNReservoir
    read: LinearReadout
    state_dim: int
    res_dim: int
    dtype: Float

    def __init__(self,
                 state_dim: int,
                 res_dim: int,
                 *,
                 key: PRNGKeyArray,
                 dtype=jax.numpy.float64
                 ) -> None:
        """Initialize reservoir and readout modules.
        
        Args:
        res (ESNReservoir): reservoir layer
        read (LinearReadout): readout layer
        state_dim (int): input/output dimension of reservoir
        res_dim (int): reservoir dimension
        key (PRNGKeyArray)
        """
        self.state_dim = state_dim
        self.res_dim = res_dim

        reskey, readkey = jax.random.split(key)
        self.res = ESNReservoir(state_dim, res_dim, key=reskey)
        self.read = LinearReadout(state_dim, res_dim, key=readkey)

        self.dtype = dtype

        
    @eqx.filter_jit
    def force(self,
              in_seq: Float[Array, "... {self.state_dim}"],
              res_state: Float[Array, "{self.res_dim}"]
              ) -> Float[Array, "... {self.res_dim}"]:
        """Teacher forces the reservoir.
        
        Args:
        in_seq (Array): input sequence to force the reservoir
            (shape=(seq_len, state_dim))
        res_state (Array): initial reservoir state (shape=(res_dim,))

        Returns:
        res_seq (Array): forced reservoir sequence (shape=(seq_len, res_dim))
        """


        def scan_fn(state, in_vars):
            res_state = self.res.advance(in_vars, state)
            return (res_state, res_state)
        
        _, res_seq = jax.lax.scan(scan_fn, res_state, in_seq)
        return res_seq
    
    @eqx.filter_jit
    def forecast(self,
                 fcast_len: int,
                 res_state: Float[Array, "{self.res_dim}"]
                 ) -> Float[Array, "fcast_len {self.state_dim}"]:
        """Forecast from an initial reservoir state.

        Args:
        fcast_len (int): steps to forecast
        res_state (Array): initial reservoir state (shape=(res_dim))

        Returns:
        seq_states: forecasted states (shape=(fcast_len, state_dim))
        """
        def scan_fn(state, _):

            out_state = self.res.advance(self.read.readout(state), state)
            return (out_state, self.read.readout(out_state))

        _, state_seq = jax.lax.scan(scan_fn, 
                                     res_state, 
                                     None, 
                                     length=fcast_len)
        return state_seq
    
def train(model: ESN,
          in_seq: Float[Array, "... {model.state_dim}"],
          res_state: Float[Array, "{model.res_dim}"],
          targets: Float[Array, "... {model.state_dim}"],
          spinup: int,
          beta: float = 8e-8
          ) -> ESN:
    """Wrapper for training a LinearReadout layer.
    
    Args:
    model (ESN): ESN to train
    in_seq (Array): forcing sequence for reservoir (shape=(seq_len, res_dim))
    res_state (Array): initial reservoir state (shape=(res_dim,))
    targets (Array): target values (shape=(seq_len, state_dim))
    spinup (int): initial transient of reservoir states to discard
    beta (float): Tikhonov regularization parameter

    Returns:
    model (ESN): trained ESN model
    """

    res_seq = model.force(in_seq, res_state)
    read = train_linear(model.read, res_seq, targets, spinup, beta)
    where = lambda m: m.read
    model = eqx.tree_at(where, model, read)
    return model