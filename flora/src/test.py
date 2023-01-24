import numpy as np

import visual
import loss
import graph
import optim
import nn



def inference(link):
    forward_probe = graph.ForwardProbe()
    out = forward_probe.trace(link)
    forward_probe.clear_cache(link)
    return out

def optimize(link):
    forward_probe = graph.ForwardProbe()
    forward_probe.trace(link)
    grad_probe = graph.GradientProbe()
    grad_probe.trace(link, None)
    optimizer = optim.StochasticGradientDescent(lr=0.01)
    optim_probe = graph.OptimizationProbe(optimizer)
    optim_probe.trace(link)

    forward_probe.clear_cache(mse)




inp = graph.Tensor(np.array([5.0,5.0,5.0]), frozen=True, des="input")
lnr = nn.Linear(inp,[2,3])
lbl = graph.Tensor(np.array([6.0,6.0]), frozen=True, des="label")
mse = loss.MeanSquaredError(lnr.link)
mse.attach(lbl)

#visual.summary(mse)

for i in range(10):
    optimize(mse)
    print(inference(mse))



'''

forward_probe = graph.ForwardProbe()
print(forward_probe.trace(mse))

grad_probe = graph.GradientProbe()
grad_probe.trace(mse, None)

optimizer = optim.StochasticGradientDescent(0.001)
optim_probe = graph.OptimizationProbe(optimizer)
optim_probe.trace(mse)

visual.summary(mse)
'''