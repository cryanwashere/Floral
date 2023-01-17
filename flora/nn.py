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
    def __init__(self, parent, shape):
        self.weight = graph.Tensor(np.random.rand(*shape) * 0.1)
        
        self.matmul = MatMul([parent, self.weight])

        self.link = self.matmul
    def __str__(self):
        return "Weight, dim: {}".format(self.weight.param.shape)

class Bias(graph.GraphModule):
    def __init__(self, parent, dim):
        self.bias = graph.Tensor(np.random.rand(dim) * 0.1)

        self.add = Add([parent, self.bias])

        self.link = self.add
    def __str__(self):
        return "Bias, dim: {}".format(self.bias.param.shape)

class Linear(graph.GraphModule):
    def __init__(self, parent, shape):

        self.weight = Weight(parent, shape)
        self.bias = Bias(self.weight.link, shape[-1])

        self.link = self.bias.link




