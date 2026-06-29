import jax
import jax.numpy as jnp


# A, B are tensors on device
@jax.jit(static_argnames=["BATCH", "M", "N", "K"])
def solve(A: jax.Array, B: jax.Array, BATCH: int, M: int, N: int, K: int) -> jax.Array:
    # return output tensor directly
    A = A.reshape((BATCH, M, K))
    B = B.reshape((BATCH, K, N))
    return jnp.einsum('...ik, ...kj -> ...ij', A, B)
