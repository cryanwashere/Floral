import graph
import datasets
import nn
import loss
import visual
import optim


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

mnist = datasets.MNIST()
sample_image, sample_label = mnist[0]
#print(sample_label.param)
model.input.attach(sample_image)
model.crossentropy.attach(sample_label)

#visual.summary(model.crossentropy)

forward_probe = graph.ForwardProbe()
sample_loss = forward_probe.trace(model.crossentropy)
print("sample loss: {}".format(sample_loss))

gradient_probe = graph.GradientProbe()
gradient_probe.trace(model.crossentropy)
print("finished computing sample gradients")

optim_probe = graph.OptimizationProbe(optim.StochasticGradientDescent(lr=0.01))
optim_probe.trace(model.crossentropy)
print("finished optimizing gradient with SGD")
graph.clear_cache(model.crossentropy)
print("cleared the cache")

print("new sample loss: {}".format(forward_probe.trace(model.crossentropy)))

