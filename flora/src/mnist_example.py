import graph
import datasets
import nn
import loss

mnist = datasets.MNIST()

class Model(graph.GraphModule):
    def __init__(self, parent):
        self.linear1 = nn.Linear(parent,[64, 768])
        self.relu1 = nn.ReLU(self.linear1.link)
        self.linear2 = nn.Linear(self.relu1, [64, 64])
        self.relu2 = nn.ReLU(self.linear2.link)
        self.linear3 = nn.Linear(self.relu2, [10,64])

        self.crossentropy = loss.CategoricalCrossEntropy(self.linear3.link)

        self.roots = [self.relu2,self.crossentropy]
