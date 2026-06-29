import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit
def solve(input: jax.Array, lo: float, hi: float, N: int) -> jax.Array:
    # return output tensor directly
    return jnp.maximum(jnp.minimum(input, jnp.ones_like(input) * hi), jnp.ones_like(input) * lo)
