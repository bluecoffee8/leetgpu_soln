import jax
import jax.numpy as jnp


# A, B are tensors on device
@jax.jit
def solve(
    A: jax.Array, B: jax.Array, C: jax.Array, M: int, N: int, K: int, alpha: float, beta: float
) -> jax.Array:
    # return output tensor directly
    return alpha * (A @ B) + beta * C
