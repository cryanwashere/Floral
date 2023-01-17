import graph
import jax.numpy as jnp
from jax import grad

class LossNode(graph.GraphNode):
    def __init__(self, parent):
        super().__init__()
        self.parents = [parent, None]

        # the loss will only have a gradient function for the 
        # input, not for the label
        grad_fn = grad(self.fn, argnums=0)
        self.grad_fns = [grad_fn]

    def attach(self, label):
        self.parents[1] = label



class MeanSquaredError(LossNode):
    def __init__(self, parent):
        super().__init__(parent)
    @staticmethod
    def fn(x,y):
        return (x - y) ** 2
    def __str__(self):
        return "Mean Squared Error Loss"
    