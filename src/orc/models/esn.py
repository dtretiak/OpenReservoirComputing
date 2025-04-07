import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Float
from orc.readouts import LinearReadout
from orc.drivers import ESNDriver
from orc.embeddings import LinearEmbedding

jax.config.update("jax_enable_x64", True)

class ESN(eqx.Module):
    """Basic implementation of ESN for forecasting.
    
    driver (ESNDriver): class determining how reservoir is driven
    readout (LinearReadout): trainable linear readout layer
    embedding (LinearEmbedding): untrainable linear embedding layer
    state_dim (int): input/output dimension
    res_dim (int): reservoir dimension
    dtype (Float): dtype of model, jnp.float64 or jnp.float32
    """
    driver: ESNDriver
    readout: LinearReadout
    embedding: LinearEmbedding
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
        self.state_dim = state_dim
        self.res_dim = res_dim

        driverkey, readoutkey, embeddingkey = jax.random.split(key, 3)
        self.driver = ESNDriver(res_dim=res_dim, key=driverkey)
        self.readout = LinearReadout(state_dim, res_dim, key=readoutkey)
        self.embedding = LinearEmbedding(state_dim, res_dim, scaling= 0.084, key=embeddingkey)

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
            proj_vars = self.embedding.embed(in_vars)
            res_state = self.driver.advance(proj_vars, state)
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
            out_state = self.driver.advance(
                self.embedding.embed(self.readout.readout(state)),
                state
                )
            return (out_state, self.readout.readout(out_state))

        _, state_seq = jax.lax.scan(scan_fn, 
                                     res_state, 
                                     None, 
                                     length=fcast_len)
        return state_seq


def train_ESN(model: ESN,
          in_seq: Float[Array, "... {model.state_dim}"],
          res_state: Float[Array, "{model.res_dim}"],
          targets: Float[Array, "... {model.state_dim}"],
          spinup: int,
          beta: float = 8e-8
          ) -> ESN:
    """Training function for ESN forecaster.
    
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
    lhs = (res_seq[spinup:].T @ res_seq[spinup:] 
            + beta * jnp.eye(model.res_dim, dtype=model.dtype))
    rhs = res_seq[spinup:].T @ targets[spinup:]
    cmat = jnp.linalg.lstsq(lhs, rhs, rcond=None)[0].T
    where = lambda m: m.readout.wout
    model = eqx.tree_at(where, model, cmat)
    return model