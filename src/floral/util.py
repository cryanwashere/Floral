import jax.numpy as jnp
import os
import graph

def save_graph_tensors(path: str, link: graph.GraphNode) -> None:
    '''
        recursively iterate through the graph, and save each
        of the tensors in a dir given by path
    '''

    # make sure that the directory where the tensors are to be saved exists
    if not os.path.isdir(path):
        os.mkdir(path)

    graph_tensors = graph.child_trace(link)
    for i, tensor in enumerate(graph_tensors):
        with open( os.path.join(path, str(i)) + ".npy", "wb" ) as f:
            if tensor is None:
                jnp.save(f, jnp.array(None))
            else:
                jnp.save(f, tensor.param)


def load_graph_tensors(path: str, link: graph.GraphNode) -> None:
    '''
        grab all of the tensors in the graph, and set them to the values 
        the tensors saved in path.
    '''
    graph_tensors = graph.child_trace()
    saved_tensor_paths = [os.path.join(path, filename) for filename in os.listdir(path)]

    for tensor, array_path in zip(graph_tensors, saved_tensor_paths):
        with open(array_path, "rb") as f:
            saved_param = jnp.load(f)
        
        if tensor is None:
            if not saved_param == None: 
                tensor = graph.Tensor(saved_param)
        elif saved_param == None:
            continue
        else:
            tensor.param = saved_param