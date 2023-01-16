import graph as g
import numpy as np
from jax import grad
import jax.numpy as jnp

def fn(x):
    return (x * 2), x
grad_fn = grad(fn)
print(grad_fn(100.0))
