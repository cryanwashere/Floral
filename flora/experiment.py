import numpy as np

import visual
import graph
import nn



_input = graph.Tensor(np.random.rand(64) * 0.1)
linear = nn.Linear(_input, 64)

forward = graph.ForwardProbe()

print(forward.trace(linear.link))

visual.summary(linear.link)