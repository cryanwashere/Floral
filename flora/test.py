import numpy as np

import visual
import loss
import graph
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

inp = graph.Tensor(np.array([5.0,5.0]))
lbl = graph.Tensor(np.array([6.0,6.0]))
mse = loss.MeanSquaredError(inp)
mse.attach(lbl)

forward_probe = graph.ForwardProbe()
print(forward_probe.trace(mse))

grad_probe = graph.GradientProbe()
grad_probe.trace(mse, None)
