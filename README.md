# Floral
![header](floral.png)

The best neural network library


Floral is a neural network library, created in Jax, by Cameron Ryan. In floral, every tensor and operation is a graph node, and graphs are both inferenced and optimized through the same probe tracing algorithm. The benefit of floral is that it's simple and efficient graph algorithm provides an easy interface with low level features. 


# installation
install with pip
```shell
pip install floral
```


# getting started

To use floral, you must create a graph by linking nodes together. Let's first define a neural network using the ```floral.graph.GraphModule``` class.

```python
from floral import nn, graph, datasets, loss, optim

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
```

When constructing a graph in floral, there exists ```floral.graph.GraphNode``` objects, and ```floral.graph.GraphModule``` objects. All of a graph's functionality comes from the ```floral.graph.GraphNode``` objects, which either store data, or perform functions, and are linked to parent nodes. The ```floral.graph.GraphModule``` objects simply contain node objects, and exist only for abstraction. All ```floral.graph.GraphModule``` objects must have a ```link``` attribute, which is a reference to the last node in their graph. 

lets load the MNIST dataset to train our nerual network on.
```python
mnist = datasets.MNIST()
```

When we want to inference our graph, we attach the variable tensors to their respective nodes, in this case the model's input node, and loss node, and use the ```floral.graph.forward_trace(node)``` method to get the node's output, which is the model's loss in this case.

 ```python
 def inference(input_link, loss_link, x, y):
    input_link.attach(x)
    loss_link.attach(y)
    out = graph.forward_trace(loss_link)
    graph.clear_cache(loss_link)
    return out
 ```
 
 lets grab a sample image, and label, and inference it on the graph

```python
sample_image, sample_label = mnist[0]
print(inference(model.input, model.crossentropy, sample_image, sample_label))
```
 
After inferencing a graph, we can use the ```floral.graph.gradient_trace(node)``` method to calculate gradients for each tensor in the graph, and then optimize them with a ```floral.graph.OptimizationProbe``` object. It is also very important to clear the graph's cache before it is traced again, through the ```floral.graph.clear_cache(node)``` method
 
 ```python
 def optimize(optim_probe, input_link, loss_link, x, y):
    input_link.attach(x)
    loss_link.attach(y)

    loss = graph.forward_trace(loss_link)
    graph.gradient_trace(loss_link)
    optim_probe.trace(loss_link)

    graph.clear_cache(loss_link)
    return loss
```

To make an optimization probe, we need a ```floral.optim.Optimizer``` object. For this, we will use ```floral.optim.StochasticGradientDescent```.

```python
optim_probe = graph.OptimizationProbe(optim.StochasticGradientDescent(lr=0.01))
```

Now lets optimize the loss on our sample image, and sample label.

```python
optimize(optim_probe, model.input, model.crossentropy, sample_image, sample_label)
print(inference(model.input, model.crossentropy, sample_image, sample_label))
```

Lets also make an evaluation function.

```python
def evaluate(test_set, input_link, loss_link):
    image_set, label_set = test_set
    total_loss = 0
    for i in range(len(image_set)):
        image, label = image_set[i], label_set[i]
        total_loss += inference(input_link, loss_link, image, label)
    return total_loss / len(image_set)
    
test_images, test_labels = mnist[:2000]
print("starting loss: ",evaluate((test_images, test_labels), model.input, model.crossentropy))
```

Now, we can train our model for one epoch. For the purposes of this tutorial, this should allow you to achieve a reasonable accuracy for your model.

```python
train_images, train_labels = mnist[2000:10000]
for i in range(len(train_images)):
    image, label = train_images[i], train_labels[i]
    optimize(optim_probe, model.input, model.crossentropy, image, label)
    if i%100 == 0:
        loss = evaluate((test_images, test_labels), model.input, model.crossentropy)
        print("step {}, loss: {}".format(i, loss))
print("final loss: {}".format(evaluate((test_images, test_labels), model.input, model.crossentropy)))
```
# contact
If you have any questions, comments, concerns, or wish to collaborate, please email [Cameron Ryan](mailto:cjryanwashere@gmail.com).

