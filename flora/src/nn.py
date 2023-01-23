import graph
from jax import grad
import jax.numpy as jnp
import numpy as np

class MatMul(graph.GraphNode):
    def __init__(self, parents):
        super().__init__()
        self.parents = parents
    @staticmethod
    def fn(x,y):
        return x @ y

class Add(graph.GraphNode):
    def __init__(self, parents):
        super().__init__()
        self.parents = parents
    @staticmethod
    def fn(x,y):
        return x + y

class Weight(graph.GraphModule):
    def __init__(self, parent, shape):
        self.weight = graph.Tensor(jnp.array(np.random.rand(*shape) * 0.01), des="weight")
        #self.weight = graph.Tensor(jnp.array([[1.,2.],[1.,2.],[1.,2.]]), "weight")

        self.matmul = MatMul([self.weight, parent])

        self.link = self.matmul
    def __str__(self):
        return "Weight, dim: {}".format(self.weight.param.shape)

class Bias(graph.GraphModule):
    def __init__(self, parent, dim):
        self.bias = graph.Tensor(jnp.array(np.random.rand(dim) * 0.01), des="bias")
        #self.bias = graph.Tensor(jnp.array([1.,2.]), "bias")

        self.add = Add([parent, self.bias])

        self.link = self.add
    def __str__(self):
        return "Bias, dim: {}".format(self.bias.param.shape)

class Linear(graph.GraphModule):
    def __init__(self, parent, shape):

        self.weight = Weight(parent, shape)
        self.bias = Bias(self.weight.link, shape[0])

        self.link = self.bias.link

class Input(graph.GraphNode):
    def __init__(self):
        super().__init__()
        self.isAttached = False
    def attach(self, parent):
        self.parents = [parent]
        self.isAttached = True
    def __str__(self):
        return "Graph Input. Attached: {}".format(self.isAttached)
    @staticmethod
    def fn(x):
        return x


class ReLU(graph.GraphNode):
    def __init__(self, parent):
        super().__init__()
        self.parents = [parent]
    @staticmethod
    def fn(x):
        return jnp.maximum(x,0)

class Softmax(graph.GraphNode):
    def __init__(self, parent):
        super().__init__()
        self.parents = [parent]
    @staticmethod
    def fn(x):
        exp = jnp.exp(x)
        return exp / jnp.sum(exp)

