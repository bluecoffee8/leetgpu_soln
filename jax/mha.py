import jax
import jax.numpy as jnp


# Q, K, V are tensors on device
@jax.jit(static_argnames=["N", "d_model", "h"])
def solve(Q: jax.Array, K: jax.Array, V: jax.Array, N: int, d_model: int, h: int) -> jax.Array:
    # return output tensor directly
    Q = Q.reshape(N, h, d_model // h).transpose(1, 0, 2) # [h, N, d_head]
    K = K.reshape(N, h, d_model // h).transpose(1, 2, 0) # [h, d_head, N]
    S = jnp.einsum("...ik, ...kj -> ...ij", Q, K) / jnp.sqrt(d_model // h)
    S -= S.max(axis=-1, keepdims=True)
    S = jnp.exp(S) 
    S /= S.sum(axis=-1, keepdims=True)
    V = V.reshape(N, h, d_model // h).transpose(1, 0, 2)
    O = jnp.einsum("...ik, ...kj -> ...ij", S, V) # [h, N, d_head]
    return O.transpose(1, 0, 2).reshape(N, -1)