import nn
import graph
import jax.numpy as jnp

def test_ReLU():
    x = graph.Tensor(jnp.array([1.,-1.,0.]))
    relu = nn.ReLU(x)

    fwd_probe = graph.ForwardProbe()
    print(fwd_probe.trace(relu))




def test_Softmax():
    x = graph.Tensor(jnp.array([1.,1.,0.]))
    softmax = nn.Softmax(x)

    fwd_probe = graph.ForwardProbe()
    print(fwd_probe.trace(softmax))
test_Softmax()