import graph
import datasets
import nn
import loss
import optim
import visual

class Model(graph.GraphModule):
    def __init__(self):
        self.input = nn.Input()
        self.linear1 = nn.Linear(self.input,[64, 784])
        self.relu1 = nn.ReLU(self.linear1.link)
        self.linear2 = nn.Linear(self.relu1, [64, 64])
        self.relu2 = nn.ReLU(self.linear2.link)
        self.linear3 = nn.Linear(self.relu2, [10,64])

        self.crossentropy = loss.CategoricalCrossEntropy(self.linear3.link)


model = Model()
optim_probe = graph.OptimizationProbe(optim.StochasticGradientDescent(lr=0.01))

mnist = datasets.MNIST()
sample_image, sample_label = mnist[0]

model.input.attach(sample_image)
model.crossentropy.attach(sample_label)

visual.summary(model.crossentropy)