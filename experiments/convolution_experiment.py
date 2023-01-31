from jax.lax import conv_general_dilated
import jax.numpy as jnp
from jax import grad


sample_input = jnp.ones((1,3,28,28))
sample_kernel= jnp.ones((1,3,3,3))


#output = conv_general_dilated(
#    sample_input,
#    sample_kernel,
#    (2,2),
#    padding="SAME"
#)

conv = lambda lhs, rhs : conv_general_dilated(
    lhs, 
    rhs,
    (2,2),
    padding="SAME"
)
scalar = lambda x : jnp.sum(x)

def fn(lhs, rhs):
    return scalar(conv(lhs, rhs))

print(grad(fn)(sample_input, sample_kernel).shape)