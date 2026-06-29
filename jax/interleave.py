import jax
import jax.numpy as jnp


# A, B are tensors on device
@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    return jnp.stack([A, B], axis=1).reshape(-1)
