import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, vjp

'''
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
'''


x = jnp.array([
    [1.,1.,1.],
    [1.,1.,1.]
])

y = jnp.array([1.,1.,1.])

z = jnp.array([0.,0.])

k = jnp.array([0.,1.])


def W(x,y):
    return x @ y
def B(W,z):
    return W + z
def L(B,k):
    return jnp.mean((B-k) ** 2)

W_out = W(x,y)
B_out = B(W_out,z)
L_out = L(B_out,k)

gradfn_W_x = jacfwd(W,0)
gradfn_B_W = jacfwd(B,0)
gradfn_L_B = jacfwd(L,0)


grad_L_B = gradfn_L_B(B_out,k)
grad_B_W = gradfn_B_W(W_out,z)
grad_W_x = gradfn_W_x(x,y)
print((grad_L_B @ grad_B_W) @ grad_W_x)