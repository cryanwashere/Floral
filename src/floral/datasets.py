from PIL import Image
from numpy import asarray, frombuffer
import graph
import numpy as np
import jax.numpy as jnp
from pkg_resources import resource_stream

class MNIST(object):
    def __init__(self):
        image = Image.open(resource_stream('floral','data/mnist_sprite.png'))
        self.data = asarray(image) / 255.0

        with resource_stream('floral','data/mnist_labels_uint8') as fp:
            integers = frombuffer(fp.read(65000 * 10), dtype='uint8')
        self.labels = integers.reshape(65000, 10)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, i):
        return graph.Tensor(jnp.array(self.data[i]), frozen=True), graph.Tensor(self.labels[i].astype(np.float64), frozen=True, des="MNIST label")
