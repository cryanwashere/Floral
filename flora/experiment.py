import numpy as np

import visual
import loss
import graph
import nn



_input = graph.Tensor(np.random.rand(64) * 0.1)
linear = nn.Linear(_input, [64,64])
linear2 = nn.Linear(linear.link, [64,1])

mse = loss.MeanSquaredError(linear2.link)
label = graph.Tensor(np.random.randint(10))
mse.attach(label)

forward = graph.ForwardProbe()
_loss = forward.trace(mse)
print("loss: {}".format(_loss))

visual.summary(mse)