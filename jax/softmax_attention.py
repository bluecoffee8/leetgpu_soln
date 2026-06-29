import jax
import jax.numpy as jnp


# Q, K, V are tensors on device
@jax.jit
def solve(Q: jax.Array, K: jax.Array, V: jax.Array, M: int, N: int, d: int) -> jax.Array:
    # return output tensor directly
    S = Q @ K.T / jnp.sqrt(d) 
    S -= S.max(axis=-1, keepdims=True) 
    S = jnp.exp(S) 
    S /= S.sum(axis=-1, keepdims=True)
    return S @ V
