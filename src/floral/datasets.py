from PIL import Image
from numpy import asarray, frombuffer
import graph
import numpy as np
import jax.numpy as jnp
from pkg_resources import resource_stream
import os
import random

class Dataset(object):
    def __init__(self):
        pass


    def __len__(self):
        return self.data.shape[0]
    
    def map(self, map_fn):
        for i in range(len(self)):
            self.data[i], self.labels[i] = map_fn(self.data[i], self.labels[i])
    
    def shuffle(self, buffer_size=64, shuffle_itr=16):
        for i in range(shuffle_itr):
            shuffle_buffer_idx = np.random.randint(len(self), size=buffer_size)
            shuffle_buffer_data, shuffle_buffer_labels = self.data[shuffle_buffer_idx], self.labels[shuffle_buffer_idx]
            shuffle_buffer_idx = np.random.shuffle(shuffle_buffer_idx)
            self.data[shuffle_buffer_idx], self.labels[shuffle_buffer_idx] = shuffle_buffer_data, shuffle_buffer_labels

            

class MNIST(object):
    def __init__(self):
        image = Image.open(resource_stream('floral','data/mnist_sprite.png'))
        self.data = asarray(image) / 255.0

        with resource_stream('floral','data/mnist_labels_uint8') as fp:
            integers = frombuffer(fp.read(65000 * 10), dtype='uint8')
        self.labels = integers.reshape(65000, 10)
    
    def __getitem__(self, i):
        return graph.Tensor(jnp.array(self.data[i]), frozen=True), graph.Tensor(self.labels[i].astype(np.float64), frozen=True, des="MNIST label")

class from_sorted_dir(Dataset):
    def __init__(self, path):
        self.class_list = os.listdir(path)

        self.data = jnp.array()
        self.labels = jnp.array()

        for i, _class in enumerate(self.class_list):
            obj_list = os.listdir(os.path.join(path, _class))
            for obj in obj_list:
                im_path = os.path.join(path, _class, obj)
                im = Image.open(im_path)
                im_data = asarray(im)
                self.data = np.append(self.data, im_data)
                self.labels = np.append(self.labels, i)

        # we now have our dataset loaded, but it is completely unsorted
    def __getitem__(self, i):
        return graph.Tensor(jnp.array(self.data[i]), frozen=True, des="dataset tensor"), graph.Tensor(self.labels[i].astype(np.float64), frozen=True, des="dataset label")


