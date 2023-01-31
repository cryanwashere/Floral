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
optim_probe = graph.OptimizationProbe(optim.StochasticGradientDescent(lr=0.01))

mnist = datasets.MNIST()
sample_image, sample_label = mnist[0]

def optimize(optim_probe, input_link, loss_link, x, y):
    input_link.attach(x)
    loss_link.attach(y)

    loss = graph.forward_trace(loss_link)
    graph.gradient_trace(loss_link)
    optim_probe.trace(loss_link)

    graph.clear_cache(loss_link)
    return loss

def inference(input_link, loss_link, x, y):
    input_link.attach(x)
    loss_link.attach(y)
    out = graph.forward_trace(loss_link)
    graph.clear_cache(loss_link)
    return out

def evaluate(test_set, input_link, loss_link):
    image_set, label_set = test_set
    total_loss = 0
    for i in range(len(image_set)):
        image, label = image_set[i], label_set[i]
        total_loss += inference(input_link, loss_link, image, label)
    return total_loss / len(image_set)

test_images, test_labels = mnist[:2000]
print("starting loss: ",evaluate((test_images, test_labels), model.input, model.crossentropy))

train_images, train_labels = mnist[2000:10000]
for i in range(len(train_images)):
    image, label = train_images[i], train_labels[i]
    optimize(optim_probe, model.input, model.crossentropy, image, label)
    if i%100 == 0:
        loss = evaluate((test_images, test_labels), model.input, model.crossentropy)
        print("step {}, loss: {}".format(i, loss))
print("final loss: {}".format(evaluate((test_images, test_labels), model.input, model.crossentropy)))
#print(inference(model.input, model.crossentropy, sample_image, sample_label))
#optimize(optim_probe, model.input, model.crossentropy, sample_image, sample_label)
#print(inference(model.input, model.crossentropy, sample_image, sample_label))
