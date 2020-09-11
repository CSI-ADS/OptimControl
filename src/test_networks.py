import networkx as nx
import numpy as np

def get_test_network(name):
    name = name.lower()
    if name == "vitali": return vitali_network()
    if name == "simple_chain": return simple_chain()
    if name == "simple_cycle": return simple_cycle()
    raise NotImplemented("name not found")

def vitali_network():
    G = nx.DiGraph()
    for u in range(6):
        G.add_node(u)
    G.add_edge(0, 1, weight=0.1)
    G.add_edge(1, 2, weight=0.5)
    G.add_edge(1, 3, weight=0.5)
    G.add_edge(1, 4, weight=0.2)
    G.add_edge(2, 1, weight=0.3)
    G.add_edge(2, 4, weight=0.2)
    G.add_edge(3, 1, weight=0.3)
    G.add_edge(3, 4, weight=0.6)
    G.add_edge(4, 1, weight=0.3)
    G.add_edge(4, 2, weight=0.5)
    G.add_edge(4, 3, weight=0.5)
    G.add_edge(4, 5, weight=1)
    return G

def simple_chain():
    G = nx.DiGraph()
    for u in range(5):
        G.add_node(u)
    G.add_edge(0, 1, weight=0.9)
    G.add_edge(1, 2, weight=0.51)
    G.add_edge(2, 3, weight=0.3)
    G.add_edge(2, 4, weight=0.9)
    return G

def simple_cycle():
    G = nx.DiGraph()
    for u in range(4):
        G.add_node(u)
    G.add_edge(0, 1, weight=0.05)
    G.add_edge(1, 2, weight=0.15)
    G.add_edge(2, 3, weight=0.15)
    G.add_edge(3, 1, weight=0.1)
    G.add_edge(1, 3, weight=0.85)
    G.add_edge(3, 2, weight=0.85)
    G.add_edge(2, 1, weight=0.85)
    return G

def simple_star():
    G = nx.full_rary_tree(5, 50, create_using=nx.DiGraph)
    return G
