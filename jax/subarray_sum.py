import jax
import jax.numpy as jnp


# input is a tensor on device
def solve(input: jax.Array, N: int, S: int, E: int) -> jax.Array:
    # return output tensor directly
    L = S 
    R = E+1
    return input[L:R].sum()
