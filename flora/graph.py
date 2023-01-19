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

        self.link = self
    def make_grad_fns(self):
        for i in range(len(self.parents)):
            # get a function that computes the derivative of the 
            # node's output, with respect to the ith parent
            grad_fn = grad(self.fn,argnums=i)
            self.grad_fns.append(grad_fn)
    #def __str__(self):
        #return "\nGraph node:\n parents:\n{}\ninput cache:\n{}\n global gradient cache:\n {}".format(self.parents, self.cache, self.global_grad_cache)

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
    def __str__(self):
        #return "Tensor {}".format(self.des)
        return "Tensor {}\nparam:\n{}\ngrad:{}\n".format(self.des,self.param,self.grad)
        

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
    

class ForwardProbe(object):
    def __init__(self):
        pass
    def trace(self, node):
        # run the node's function on the parent's output values
        if node.isTensor:
            return node.param
        else:
            # check if the node has already gathered it's value
            if node.cache is None:
                # recursively gather the output values from the node's
                # parents, then store the output values in the node's 
                # cache for differentiation
                node.cache = list()
                for parent in node.parents:
                    node.cache.append(self.trace(parent))
            out = node.fn(*node.cache)
            #print(out)
            return out
    def clear_cache(self, node):
        if node.cache is not None:
            for parent in node.parents:
                self.clear_cache(parent)
            node.cache = None

class GradientProbe(object):
    def __init__(self):
        pass
    def trace(self, node, dL):
        
        # the node should have a jax-computed gradient function with
        # respect to every one of it's parents

        if node.isFork:
            # this will manage backwards differentiation for fork nodes

            # count how many times the probe has reached the fork. 
            node.fork_counter_cache += 1
            # compute the gradient for the fork node
            if node.fork_cache == None:
                node.fork_cache = dL
            else:
                node.fork_cache += dL
            # if the probe has reached the fork as many times as it splits,
            # then it the probe will continue tracing
            if node.fork_counter_cache == node.fork_count:
                self.trace(node.parents[0])

        elif node.isTensor:

            node.grad = dL

        else:
            node.local_grad_cache = list()
            # compute the derivative of the node with respect to each of the parents
            for i, cache in enumerate(node.cache):
                jacfwd_grad_fn = jacfwd(node.fn, argnums=i)
                node.local_grad_cache.append(jacfwd_grad_fn(*node.cache))

            
            #print(node, node.local_grad_cache)
            #compute the global gradients
            node.global_grad_cache = list()
            for grad_cache in node.local_grad_cache:
                #print(node, grad_cache, dL)
                if dL is None:
                    node.global_grad_cache.append(grad_cache)
                else:
                    # chain rule

                    # TODO
                    # it needs to be determined whether to do 
                    # regular multiplcation ( * ), or matrix
                    # multiplication ( @ )
                    #print(node)
                    if (len(grad_cache.shape) >= 1) and (len(dL.shape) >= 1):
                    #    print("@ grad: {}, dL: {}".format(grad_cache.shape, dL.shape))
                        node.global_grad_cache.append(dL @ grad_cache)
                    else:
                    #    print("* grad: {}, dL: {}".format(grad_cache.shape, dL.shape))
                        node.global_grad_cache.append(grad_cache * dL)

            
            # recursively backpropogate through the graph
            for i, parent in enumerate(node.parents):
                self.trace(parent, node.global_grad_cache[i])
    def clear_cache(self, node):
        if node.isFork:
            node.fork_cache = None
            node.fork_counter_cache = None
        if node.local_grad_cache is not None:
            node.local_grad_cache = None
        if node.global_grad_cache is not None:
            node.global_grad_cache = None
        if node.grad is not None:
            node.grad = None
        if node.cache is not None:
            node.cache = None
            for parent in node.parents:
                self.clear_cache(parent)
        

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
            for parent in node.parents:
                self.trace(parent)
