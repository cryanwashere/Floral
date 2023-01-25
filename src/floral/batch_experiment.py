import graph
import nn
import loss
import jax.numpy as jnp
from jax import vmap, grad

a = graph.Tensor(jnp.ones((3,784)))
b = graph.Tensor(jnp.ones((784,2)))
c = graph.Tensor(jnp.array([
    [0.,1,],
    [0.,1.],
    [0.,1.]
]))

mm = nn.MatMul([a,b])
ls = loss.MeanSquaredError(mm)
ls.attach(c)

print(graph.forward_trace(ls))

graph.gradient_trace(ls)
print(a.grad.shape)