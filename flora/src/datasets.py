from PIL import Image
from numpy import asarray, frombuffer
import graph

class MNIST(object):
    def __init__(self):
        image = Image.open('utils/mnist_sprite.png')
        self.data = asarray(image) / 255.0

        with open ('utils/mnist_labels_uint8','rb') as fp:
            integers = frombuffer(fp.read(65000 * 10), dtype='uint8')
        self.labels = integers.reshape(65000, 10)

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, i):
        return graph.Tensor(self.data[i]), graph.Tensor(self.labels[i])
