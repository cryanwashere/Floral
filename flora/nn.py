import graph
from jax import grad
import jax.numpy as jnp
import numpy as np

class MatMul(graph.GraphNode):
    def __init__(self, parents):
        super().__init__()
        self.parents = parents
        self.make_grad_fns()
    @staticmethod
    def fn(x,y):
        return x @ y

class Add(graph.GraphNode):
    def __init__(self, parents):
        super().__init__()
        self.parents = parents
        self.make_grad_fns()
    @staticmethod
    def fn(x,y):
        return x + y


class Weight(graph.GraphModule):
    def __init__(self, parent, dim):
        self.weight = graph.Tensor(np.random.rand(dim,dim) * 0.1)
        
        self.matmul = MatMul([parent, self.weight])

        self.link = self.matmul

class Bias(graph.GraphModule):
    def __init__(self, parent, dim):
        self.bias = graph.Tensor(np.random.rand(dim) * 0.1)

        self.add = Add([parent, self.bias])

        self.link = self.add

class Linear(graph.GraphModule):
    def __init__(self, parent, dim):

        self.weight = Weight(parent, dim)
        self.bias = Bias(self.weight.link, dim)

        self.link = self.bias.link




_input = graph.Tensor(np.random.rand(64) * 0.1)
linear = Linear(_input, 64)

forward = graph.ForwardProbe()

print(forward.trace(linear.link))