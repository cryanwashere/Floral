import jax.numpy as jnp
import os
import graph

def save_graph_tensors(path: str, link: graph.GraphNode) -> None:
    '''
        recursively iterate through the graph, and save each
        of the tensors in a dir given by path
    '''
    if not os.path.isdir(path):
        os.mkdir(path)
    

    graph_tensors = graph.child_trace(link)


def load_graph_tensors(path: str, link: object) -> None:
    pass