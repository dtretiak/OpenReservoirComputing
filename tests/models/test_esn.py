import orc
import jax

def test_esn_train():
    esn = orc.models.ESN(state_dim=3, res_dim=100, key=jax.random.key(0))
    esn = orc.models.esn.train(esn,
                               jax.random.normal(key=jax.random.key(0),shape=(300,3)),
                               jax.numpy.ones(100),
                               jax.random.normal(key=jax.random.key(3),shape=(300,3),), 
                               spinup=3,
                               beta=5e-7)
    assert jax.numpy.sum(esn.read.w_or) != 0