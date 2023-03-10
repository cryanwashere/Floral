from .src.floral import graph

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
image_batch, label_batch = mnist[:64]

print(image_batch.shape())

model.input.attach(image_batch)
model.crossentropy.attach(label_batch)

print(graph.forward_trace(model.crossentropy))