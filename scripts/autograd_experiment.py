import jax.numpy as jnp
from jax import jacfwd, grad
import graph
from optim import StochasticGradientDescent


class Fn(graph.GraphNode):
    def __init__(self,parents):
        super().__init__()
        self.parents = parents
        
    @staticmethod   
    def fn(x,y):
        return x @ y

class L(graph.GraphNode):
    def __init__(self, parent):
        super().__init__()
        self.parents = [parent]  
        
    @staticmethod     
    def fn(f):
        return jnp.sum(f)

x = graph.Tensor(jnp.ones((64,784)))
y = graph.Tensor(jnp.ones((784)))
fn = Fn([x,y])
l = L(fn)


forward_probe = graph.ForwardProbe()
forward_probe.trace(l)

class GradientProbe(object):
    def trace(self, node, phi=None):
        if phi is None:
            phi = lambda x : x
        for i, parent in enumerate(node.parents):
            psi = lambda y: phi(node.fn( *node.cache[:i], y, *node.cache[i+1:] ))
            if parent.isTensor:
                parent.grad = grad(psi)(node.cache[i])
            self.trace(parent, psi)
gradient_probe = GradientProbe()
gradient_probe.trace(l)
optimization_probe = graph.OptimizationProbe(StochasticGradientDescent(lr=0.01))
optimization_probe.trace(l)
graph.clear_cache(l)
print(forward_probe.trace(l))

for i in range(10):
    gradient_probe.trace(l)
    optimization_probe.trace(l)
    graph.clear_cache(l)
    print(forward_probe.trace(l))