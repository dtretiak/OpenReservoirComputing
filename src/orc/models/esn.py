"""Classic ESN implementation with tanh nonlinearity and linear readout."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from orc.drivers import ESNDriver
from orc.embeddings import LinearEmbedding
from orc.readouts import LinearReadout

jax.config.update("jax_enable_x64", True)


class ESN(eqx.Module):
    """Basic implementation of ESN for forecasting.

    Attributes
    ----------
    driver : ESNDriver
        Class determining how reservoir is driven.
    readout : LinearReadout
        Trainable linear readout layer.
    embedding : LinearEmbedding
        Untrainable linear embedding layer.
    state_dim : int
        Input/output dimension.
    res_dim : int
        Reservoir dimension
    dtype : Float
        Dtype of model, jnp.float64 or jnp.float32.

    Methods
    -------
    force(in_seq, res_state)
        Teacher force the reservoir with sequence in_seq and init. cond. res_state.
    forecast(fcast_len, res_state)
        Perform a forecast of fcast_len steps from res_state.
    """

    driver: ESNDriver
    readout: LinearReadout
    embedding: LinearEmbedding
    state_dim: int
    res_dim: int
    dtype: Float

    def __init__(
        self,
        state_dim: int,
        res_dim: int,
        *,
        key: PRNGKeyArray,
        dtype=jax.numpy.float64,
    ) -> None:
        self.state_dim = state_dim
        self.res_dim = res_dim

        driverkey, readoutkey, embeddingkey = jax.random.split(key, 3)
        self.driver = ESNDriver(res_dim=res_dim, key=driverkey)
        self.readout = LinearReadout(state_dim, res_dim, key=readoutkey)
        self.embedding = LinearEmbedding(
            state_dim, res_dim, scaling=0.084, key=embeddingkey
        )

        self.dtype = dtype

    @eqx.filter_jit
    def force(self, in_seq: Array, res_state: Array) -> Array:
        """Teacher forces the reservoir.

        Parameters
        ----------
        in_seq: Array
            Input sequence to force the reservoir, (shape=(seq_len, state_dim)).
        res_state : Array
            Initial reservoir stat, (shape=(res_dim,)).

        Returns
        -------
        Array
            Forced reservoir sequence, (shape=(seq_len, res_dim)).
        """

        def scan_fn(state, in_vars):
            proj_vars = self.embedding.embed(in_vars)
            res_state = self.driver.advance(proj_vars, state)
            return (res_state, res_state)

        _, res_seq = jax.lax.scan(scan_fn, res_state, in_seq)
        return res_seq

    @eqx.filter_jit
    def forecast(self, fcast_len: int, res_state: Array) -> Array:
        """Forecast from an initial reservoir state.

        Parameters
        ----------
        fcast_len : int
            Steps to forecast.
        res_state : Array
            Initial reservoir state, (shape=(res_dim)).

        Returns
        -------
        Array
            Forecasted states, (shape=(fcast_len, state_dim))
        """

        def scan_fn(state, _):
            out_state = self.driver.advance(
                self.embedding.embed(self.readout.readout(state)), state
            )
            return (out_state, self.readout.readout(out_state))

        _, state_seq = jax.lax.scan(scan_fn, res_state, None, length=fcast_len)
        return state_seq

    def set_readout(self, readout: LinearReadout):
        """Replace readout layer.

        Parameters
        ----------
        readout : LinearReadout
            New readout layer.

        Returns
        -------
        ESN
            Updated model with new readout layer.
        """

        def where(m: ESN):
            return m.readout

        new_model = eqx.tree_at(where, self, readout)
        return new_model

    def set_embedding(self, embedding: LinearEmbedding):
        """Replace embedding layer.

        Parameters
        ----------
        embedding : LinearEmbedding
            New embedding layer.

        Returns
        -------
        ESN
            Updated model with new embedding layer.
        """

        def where(m: ESN):
            return m.embedding

        new_model = eqx.tree_at(where, self, embedding)
        return new_model


def train_esn(
    model: ESN,
    in_seq: Array,
    res_state: Array,
    targets: Array,
    spinup: int,
    beta: float = 8e-8,
) -> ESN:
    """Training function for ESN forecaster.

    Parameters
    ----------
    model : ESN
        ESN model to train.
    in_seq : Array
        Forcing sequence for reservoir, (shape=(seq_len, res_dim)).
    res_state : Array
        Initial reservoir state, (shape=(res_dim,)).
    targets : Array
        Target values, (shape=(seq_len, state_dim)).
    spinup : int
        Initial transient of reservoir states to discard.
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    model : ESN
        Trained ESN model.
    """
    res_seq = model.force(in_seq, res_state)
    lhs = res_seq[spinup:].T @ res_seq[spinup:] + beta * jnp.eye(
        model.res_dim, dtype=model.dtype
    )
    rhs = res_seq[spinup:].T @ targets[spinup:]
    cmat = jax.scipy.linalg.solve(lhs, rhs).T

    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)
    return model
