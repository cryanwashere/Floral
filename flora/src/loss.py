import graph
import jax.numpy as jnp
from jax import grad

class LossNode(graph.GraphNode):
    def __init__(self, parent):
        super().__init__()
        self.parents = [parent, None]

    def attach(self, label):
        self.parents[1] = label



class MeanSquaredError(LossNode):
    def __init__(self, parent):
        super().__init__(parent)
    @staticmethod
    def fn(x,y):
        return jnp.mean((x - y) ** 2)
    def __str__(self):
        return "Mean Squared Error Loss"



class CategoricalCrossEntropy(LossNode):
    def __init__(self, parent):
        super().__init__(parent)
    @staticmethod
    def fn(x,y):
        #print("input",x)
        smx = jnp.exp(x)
        #print("exp",smx)
        smx = smx / jnp.sum(smx)
        #print("smx",smx)
        return -jnp.sum(y * jnp.log(smx))
    