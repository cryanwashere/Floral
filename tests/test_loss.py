import loss
import graph
import jax.numpy as jnp

def test_CategoricalCrossEntropy():
    x = graph.Tensor(jnp.array([1.5,3.4,0.1]))
    y = graph.Tensor(jnp.array([0.,0.,1.]))

    crossentropy = loss.CategoricalCrossEntropy(x)
    crossentropy.attach(y)

    fwd_probe = graph.ForwardProbe()

    print(fwd_probe.trace(crossentropy))

test_CategoricalCrossEntropy()