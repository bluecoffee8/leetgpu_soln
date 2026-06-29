import jax
import jax.numpy as jnp


# X is a tensor on device
@jax.jit
def solve(X: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    return 1.0 / (1.0 + jnp.exp(-X))
