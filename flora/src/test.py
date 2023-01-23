import numpy as np

import visual
import loss
import graph
import optim
import nn


'''
_input = graph.Tensor(np.random.rand(64) * 0.1, "input")
linear = nn.Linear(_input, [64,64])
linear2 = nn.Linear(linear.link, [64,1])

mse = loss.MeanSquaredError(linear2.link)
label = graph.Tensor(np.array([5.0]), "label")
mse.attach(label)

forward = graph.ForwardProbe()
_loss = forward.trace(mse)
print("loss: {}".format(_loss))


grad_probe = graph.GradientProbe()
grad_probe.trace(mse, None)

'''

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