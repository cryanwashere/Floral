import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np

class GraphNode(object):
    def __init__(self):
        self.cache = None
        self.isTensor = False
        self.isFork = False
        self.parents = list()
        self.child_nodes = list()
        self.link = self

        self.child_trace_cache = False

def child_trace(node):
    '''
    iterate though a graph, and attach each node to it's child nodes
    returns a list of all of the tensors in the graph
    '''
    graph_tensors = list()
    node.child_trace_cache = True
    for parent in node.parents:
        parent.child_nodes.append(node)
        if node.isTensor:
            graph_tensors.append(node)
        if not parent.child_trace_cache:
            graph_tensors += child_trace(parent)
    return graph_tensors

def fork_trace(node):
    for parent in node.parents:
        if parent.isTensor:
            return False
        if parent.child_nodes > 1:
            return True
    return False
    
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

        self.fn_cache = None

        self.fork_cache = None
        self.trace_cache = False
    @staticmethod
    def fn(x):
        return x

def forward_trace( node ):
    if node.isTensor:
        return node.param
    elif node.isFork:
        if node.fork_cache is not None:
            node.fork_cache = forward_trace(node.parents[0])
        return node.fork_cache
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
        gradient_trace(parent, phi=psi)    


class OptimizationProbe(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def trace(self, node):

        if node.isTensor:
            if not node.frozen:
                node.param = self.optimizer.optimize(node.param, node.grad)
        elif node.isFork:
            if not node.trace_cache:
                self.trace(node.parents[0])
            node.trace_cache = True
        else:
            for parent in node.parents:
                self.trace(parent)


def clear_cache(node):
        if node.isFork:
            node.fork_cache = None
            node.fork_counter_cache = None
        if node.isTensor:
            if node.grad is not None:
                node.grad = None
        if node.cache is not None:
            node.cache = None
            for parent in node.parents:
                clear_cache(parent)

def inference(link):
    out = forward_trace(link)
    clear_cache(link)
    return out

'''
def get_f_xh(link, tensor, idx, epsilon):
    tensor.param[idx] += epsilon
    return inference(link)


def check_numeric_gradient(node, epsilon=0.0001):
    
    # numerically finds the derivative of each number
    # in each tensor with respect to the output node
    
    f_x = inference(node)
    graph_tensors = child_trace(node)
    for tensor in graph_tensors:
        numeric_grad = jnp.zeros(*tensor.shape())

        
'''