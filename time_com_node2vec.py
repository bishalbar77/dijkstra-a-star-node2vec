import networkx as nx
from node2vec import Node2Vec
import numpy
import time

# Create a large graph
nodeVecGraph = nx.Graph()

# Add nodes
for i in range(10000):
    nodeVecGraph.add_node(i)

# Add edges
for i in range(10000):
    for j in range(i + 1, 1000):
        nodeVecGraph.add_edge(i, j, weight=abs(j - i))

# Use node2vec to generate node embeddings
node2vec = Node2Vec(nodeVecGraph, dimensions=128, walk_length=30, num_walks=200, workers=4)

# Learn embeddings
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get the embeddings for all nodes
embeddings = {node: model.wv[node] for node in nodeVecGraph.nodes}

start_time = time.time()
# Example: Get the embedding for a specific node
node_id = 0
embedding = embeddings[node_id]
# print(f"Embedding for node {node_id}: {embedding}")
end_time = time.time()
node2vec_time = end_time - start_time
print("Node2Vec time:", node2vec_time)