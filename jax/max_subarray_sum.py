import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit(static_argnames=['window_size'])
def solve(input: jax.Array, N: int, window_size: int) -> jax.Array:
    # return output tensor directly
    return jnp.convolve(input, jnp.ones((window_size, )), mode="valid").max()
