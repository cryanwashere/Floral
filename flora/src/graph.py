import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np

class GraphNode(object):
    def __init__(self):
        self.cache = None
        self.local_grad_cache = None
        self.global_grad_cache = None

        # There are certain types of special nodes that are very
        # important to the graph algorithm's functioning. Probes
        # will nead to know if a node is one of the special types 
        # of nodes

        # if a node is a tensor, then it does not have any parents
        # and it has a gradient variable 
        self.isTensor = False
        # if a node is a ForkNode, then that means the graph splits 
        # at it, and it needs to be treated specially
        self.isFork = False

        self.parents = list()
        self.grad_fns = list()

        self.tip = self
        self.link = self
   
    
class GraphModule(object):
    def __init__(self):
        pass

class Tensor(GraphNode):
    def __init__(self, param, frozen=False, des=""):
        super().__init__()
        self.isTensor = True
        self.param = param
        self.grad = None
        
        self.des = des
        self.frozen = frozen
    def shape(self):
        return self.param.shape
    def __str__(self):
        return "Tensor {}, shape:{}".format(self.des, self.param.shape)
    def __len__(self):
        return self.param.shape[0]
    def __getitem__(self, idx):
        return Tensor(self.param[idx], frozen=self.frozen, des=self.des)

  
        

class ForkNode(GraphNode):
    def __init__(self, parent, fork_count):
        super.__init__(self)
        self.isFork = True
        self.parents = [parent]
        self.fork_count = fork_count
        self.fork_counter_cache = 0
        self.fork_cache = None
    @staticmethod
    def fn(x):
        return x
    


def forward_trace( node ):
    if node.isTensor:
        return node.param
    else:
        node.cache = list()
        for parent in node.parents:
            node.cache.append(forward_trace(parent))
        out = node.fn(*node.cache)
        return out


def gradient_trace(node, phi=None):
    if phi is None:
        phi = lambda x : x
    for i, parent in enumerate(node.parents):
        psi = lambda y: phi(node.fn( *node.cache[:i], y, *node.cache[i+1:] ))
        if parent.isTensor:
            if not parent.frozen:
                parent.grad = grad(psi)(node.cache[i])
        gradient_trace(parent, psi)    
        

class OptimizationProbe(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def trace(self, node):
        # TODO
        # this algorithm needs to account for fork nodes
        if node.isTensor:
            #print(node)
            if not node.frozen:
                # the only thing a tensor does is store a param value, 
                # so its global gradient cache will only have one element
                #print(node)
                node.param = self.optimizer.optimize(node.param, node.grad)
        else:
            if node.isFork:
                # this will manage backwards differentiation for fork nodes

                # count how many times the probe has reached the fork. 
                node.fork_counter_cache += 1
                
                # if the probe has reached the fork as many times as it splits,
                # then it the probe will continue tracing
                if node.fork_counter_cache == node.fork_count:
                    self.trace(node.parents[0])
            else:
                for parent in node.parents:
                    self.trace(parent)

def clear_cache(node):
        if node.isFork:
            node.fork_cache = None
            node.fork_counter_cache = None
        if node.local_grad_cache is not None:
            node.local_grad_cache = None
        if node.global_grad_cache is not None:
            node.global_grad_cache = None
        if node.isTensor:
            if node.grad is not None:
                node.grad = None
        if node.cache is not None:
            node.cache = None
            for parent in node.parents:
                clear_cache(parent)

