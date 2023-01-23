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
    '''
    @staticmethod
    def fn(x):
        return x
    '''
    #def __str__(self):
        #return "\nGraph node:\n parents:\n{}\ninput cache:\n{}\n global gradient cache:\n {}".format(self.parents, self.cache, self.global_grad_cache)

class GraphModule(object):
    # this is not a graph node, but a container to store graph nodes
    def __init__(self):
        self.roots = list()
    def __call__(self, root):
        link = self.roots[root]
        forward_probe = ForwardProbe()
        out = forward_probe.trace(link)
        forward_probe.clear_cache(link)
        return out

class Tensor(GraphNode):
    def __init__(self, param, frozen=False, des=""):
        super().__init__()
        self.isTensor = True
        self.param = param
        self.grad = None
        
        self.des = des
        self.frozen = frozen
    def __str__(self):
        return "Tensor {}, shape:{}".format(self.des, self.param.shape)
        #return "Tensor {}\nparam:\n{}\ngrad:{}\n".format(self.des,self.param,self.grad)

    def __add__(self,x):
        print(self.param, x.param)
        self.param += x.param
        return self
    def __mul__(self,x):
        self.param *= x.param
        return self
    def __div__(self,x):
        self.param /= x.param
        return self
        

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
    def trace(self, node, test=False):

        if node.isTensor:
            return node.param
        else:
            node.cache = list()
            for parent in node.parents:
                node.cache.append(self.trace(parent))
            #print(node, node.cache)
            out = node.fn(*node.cache)
            return out


class GradientProbe(object):
    def __init__(self):
        pass
    def trace(self, node, phi=None):
        if phi is None:
            phi = lambda x : x
        for i, parent in enumerate(node.parents):
            psi = lambda y: phi(node.fn( *node.cache[:i], y, *node.cache[i+1:] ))
            if parent.isTensor:
                if not parent.frozen:
                    parent.grad = grad(psi)(node.cache[i])
            self.trace(parent, psi)    
        

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

'''
class GradientProbe(object):
    def __init__(self):
        pass
    def trace(self, node, dL=None):
        #print("opening node, {}".format(node))
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
               # print("iterating through node cache: {}".format(i))
                jacfwd_grad_fn = jacfwd(node.fn, argnums=i)
                #print("calculated gradient function with jax")
                #print(node.cache)
                node.local_grad_cache.append(jacfwd_grad_fn(*node.cache))
            #print(node.local_grad_cache)
            
            #print(node, node.local_grad_cache)
            #compute the global gradients
            node.global_grad_cache = list()
            for grad_cache in node.local_grad_cache:
                #print(node, grad_cache, dL)
                #print("iterating through local grad cache")
                if dL is None:
                    node.global_grad_cache.append(grad_cache)
                else:
                    # chain rule

                    # TODO
                    # it needs to be determined whether to do 
                    # regular multiplcation ( * ), or matrix
                    # multiplication ( @ )
                    #print(node)
                    #print("doing matmul")
                    if (len(grad_cache.shape) >= 1) and (len(dL.shape) >= 1):
                    #    print("@ grad: {}, dL: {}".format(grad_cache.shape, dL.shape))
                        node.global_grad_cache.append(dL @ grad_cache)
                    else:
                    #    print("* grad: {}, dL: {}".format(grad_cache.shape, dL.shape))
                        node.global_grad_cache.append(grad_cache * dL)
                    #print("done with matmul")

            #print("done with gradient iteration")
            # recursively backpropogate through the graph
            for i, parent in enumerate(node.parents):
                #print("gradient probe parent search")
                self.trace(parent, node.global_grad_cache[i])
            #print("gradient probe terminated")
'''