from PIL import Image
from numpy import asarray, frombuffer
from src.floral import graph
import numpy as np
import jax.numpy as jnp
from pkg_resources import resource_stream
import os

class Dataset(object):
    def __init__(self):
        pass


    def __len__(self):
        return self.data.shape[0]
    
    def map(self, map_fn):
        for i in range(len(self)):
            arr, lb = map_fn(self.data[i], self.labels[i])
            self.data.at[i].set(arr)
            self.labels.at[i].set(lb) 
    
    def shuffle(self, buffer_size=64, shuffle_itr=16):
        for i in range(shuffle_itr):
            shuffle_buffer_idx = np.random.randint(len(self), size=buffer_size)
            shuffle_buffer_data, shuffle_buffer_labels = self.data[shuffle_buffer_idx], self.labels[shuffle_buffer_idx]
            shuffle_buffer_idx = np.random.shuffle(shuffle_buffer_idx)
            self.data = self.data.at[shuffle_buffer_idx].set(shuffle_buffer_data)
            self.labels = self.labels.at[shuffle_buffer_idx].set(shuffle_buffer_labels)
    
    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(os.path.join(path, "data.npy"), "wb") as f:
            jnp.save(f, self.data)
        with open(os.path.join(path, "labels.npy"), "wb") as f:
            jnp.save(f, self.labels)

    def __getitem__(self, i):
        return graph.Tensor(jnp.array(self.data[i]), frozen=True, des="dataset tensor"), graph.Tensor(self.labels[i].astype(np.float64), frozen=True, des="dataset label")

            

class MNIST(object):
    def __init__(self):
        image = Image.open(resource_stream('floral','data/mnist_sprite.png'))
        self.data = asarray(image) / 255.0

        with resource_stream('floral','data/mnist_labels_uint8') as fp:
            integers = frombuffer(fp.read(65000 * 10), dtype='uint8')
        self.labels = integers.reshape(65000, 10)
    
    def __getitem__(self, i):
        return graph.Tensor(jnp.array(self.data[i]), frozen=True), graph.Tensor(self.labels[i].astype(np.float64), frozen=True, des="MNIST label")

def one_hot(index, max):
    #print(index, max)
    x = jnp.zeros(max)
    x = x.at[index].set(1.0)
    return x

class from_sorted_dir(Dataset):
    def __init__(self, path):
        self.class_list = os.listdir(path)

        self.data = None
        self.labels = None

        for i, _class in enumerate(self.class_list):
            obj_list = os.listdir(os.path.join(path, _class))
            print("opening class {}: {}, containing {} files".format(i, _class, len(obj_list)))
            
            for j, obj in enumerate(obj_list[:100]):
                im_path = os.path.join(path, _class, obj)
                im = Image.open(im_path)
                im = im.resize((256,256))
                im_data = asarray(im)
                im_data = np.expand_dims(im_data, axis=0)
                if self.data is None:
                    self.data = im_data
                else:
                    self.data = np.concatenate((self.data,im_data), axis=0)

                label = np.expand_dims(one_hot(i, len(self.class_list)), axis=0)

                if self.labels is None:
                    self.labels = label
                else:
                    self.labels = np.concatenate((self.labels, label), axis=0)

        # we now have our dataset loaded, but it is completely unsorted
    


class from_saved_npy(Dataset):
    def __init__(self, path):
        with open(os.path.join(path, "data.npy"), "rb") as f:
            self.data = jnp.load(f)
        with open(os.path.join(path, "labels.npy"), "rb") as f:
            self.labels = jnp.load(f)