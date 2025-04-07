"""Implementations of common reservoir forms. """

from orc.reservoirs.base import ReservoirBase
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jaxtyping import Array, PRNGKeyArray, Float

jax.config.update("jax_enable_x64", True)

class ESNReservoir(ReservoirBase):
    """Standard implementation of ESN reservoir with tanh nonlinearity.

    Attributes:
    in_dim (int): reservoir input dimension
    res_dim (int): reservoir dimension
    w_rr (Array): reservoir update matrix (shape=(res_dim, res_dim,))
    w_ri (Array): input matrix (shape=(res_dim, in_dim,))
    alpha (float): leak rate parameter
    rho_sr (float): spectral radius of w_rr
    rho_A (float): density of w_rr
    sigma (float): entries of w_ri drawn from U(sigma, sigma)
    sigma_b (float): additive bias in tanh nonlinearity
    dtype (Float): dtype, default jnp.float64

    Methods:
    advance(in_vars, res_state) -> updated reservoir state
    """
    in_dim: int
    res_dim: int
    alpha: float
    rho_sr: float
    rho_w: float
    sigma: float
    sigma_b: float
    dtype: Float
    w_rr: Array
    w_ri: Array
    dtype: Float

    def __init__(self,
                 in_dim: int,
                 res_dim: int,
                 alpha: float = 0.6,
                 rho_sr: float = 0.8,
                 rho_w: float = 0.02,
                 sigma: float = 0.084,
                 sigma_b: float = 1.6,
                 dtype: Float = jnp.float64,
                 *,
                 key: PRNGKeyArray,
                 ) -> None:
        """Initialize weight matrices.

        Arguments:
        in_dim (int): reservoir input dimension
        res_dim (int): reservoir dimension
        alpha (float): leak rate parameter
        rho_sr (float): spectral radius of w_rr
        rho_w (float): density of w_rr
        sigma (float): entries of w_ri drawn from U(-sigma, sigma)
        sigma_b (float): additive bias in tanh nonlinearity
        dtype (Float): dtype for model
        key (PRNGKeyArray): random seed 
        """
        super().__init__()
        self.in_dim = in_dim
        self.res_dim = res_dim
        self.alpha = alpha
        self.rho_sr = rho_sr
        self.rho_w = rho_w
        self.sigma = sigma
        self.sigma_b = sigma_b
        self.dtype = dtype
        w_rrkey1, w_rrkey2, w_rikey = jax.random.split(key, 3)

        N_nonzero = int(res_dim ** 2 * rho_w)
        w_rr_indices = jax.random.choice(w_rrkey1,
                                      res_dim ** 2,
                                      shape=(N_nonzero,),
                                      )
        w_rr_vals = jax.random.uniform(w_rrkey2,
                                    shape=N_nonzero,
                                    minval=-1,
                                    maxval=1,
                                    dtype=self.dtype
                                    )
        w_rr = jnp.zeros(self.res_dim*self.res_dim, dtype=dtype)
        w_rr = w_rr.at[w_rr_indices].set(w_rr_vals)
        w_rr = w_rr.reshape(res_dim, res_dim)
        w_rr = w_rr * (rho_sr / jnp.max(jnp.abs(jnp.linalg.eigvals(w_rr))))
        self.w_rr = sparse.BCOO.fromdense(w_rr)
        self.w_ri = jax.random.uniform(w_rikey,
                                       (res_dim, in_dim),
                                       minval=-sigma,
                                       maxval=sigma,
                                       dtype=dtype)
        
        self.dtype = dtype
        
    def advance(self,
                in_vars: Float[Array, "{self.in_dim}"],
                res_state: Float[Array, "{self.res_dim}"]
                ) -> Float[Array, "{self.res_dim}"]:       
        """Advance the reservoir state.

        Arguments:
        in_vars (Array): reservoir inputs (shape=(in_dim,))
        res_state (Array): reservoir state (shape=(res_dim,))

        Returns:
        res_next (Array): reservoir state (shape=(res_dim,))
        """
        res_next = jnp.tanh(self.w_rr @ res_state
                            + self.w_ri @ in_vars
                            + self.sigma_b * jnp.ones(self.res_dim))
        res_next = self.alpha * res_next + (1-self.alpha) * res_state
        return res_next