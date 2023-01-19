import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, vjp


W = jnp.array([
    [1.,1.,1.],
    [1.,1.,1.]
])

X = jnp.array([1.,2.,3.])



def fn(w,x):
    return w @ x



jacfwd_grad_fn = jacfwd(fn, argnums=0)
W_jacobian = jacfwd_grad_fn(W,X)

Z = fn(W,X)

def L(z):
    return jnp.sum(z)

jacfwd_grad_L = jacfwd(L)
W_grad = jacfwd_grad_L(Z) @ W_jacobian
print(L(fn(W,X)))
W = W - 0.1 * W_grad
print(L(fn(W,X)))