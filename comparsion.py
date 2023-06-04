import time
import networkx as nx
from node2vec import Node2Vec


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def add_node(self, value):
        self.nodes.add(value)
        self.edges[value] = {}

    def add_edge(self, from_node, to_node, cost):
        self.edges[from_node][to_node] = cost
        self.edges[to_node][from_node] = cost

GRAPH_NODES = 20000
GRAPH_EDGES = 20000

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        current_distance, current_node = queue.pop(0)

        if current_distance > distances[current_node]:
            continue

        for neighbor, cost in graph.edges[current_node].items():
            distance = current_distance + cost
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                queue.append((distance, neighbor))

    return distances


def heuristic(node, goal):
    return abs(node - goal)


def a_star(graph, start, goal):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        current_distance, current_node = queue.pop(0)

        if current_distance > distances[current_node]:
            continue

        if current_node == goal:
            break

        for neighbor, cost in graph.edges[current_node].items():
            distance = current_distance + cost
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                priority = distance + heuristic(neighbor, goal)
                queue.append((priority, neighbor))
        
        queue.sort(key=lambda x: x[0])

    return distances[goal]

def node2Vec():
    # Create a large graph
    nodeVecGraph = nx.Graph()

    # Add nodes
    for i in range(GRAPH_NODES):
        nodeVecGraph.add_node(i)

    # Add edges
    for i in range(GRAPH_EDGES):
        for j in range(i + 1, GRAPH_NODES):
            nodeVecGraph.add_edge(i, j, weight=abs(j - i))

    # Use node2vec to generate node embeddings
    node2vec = Node2Vec(nodeVecGraph, dimensions=128, walk_length=30, num_walks=200, workers=4)

    # Learn embeddings
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Get the embeddings for all nodes
    embeddings = {node: model.wv[node] for node in nodeVecGraph.nodes}
    return embeddings

# Create a large graph
graph = Graph()

# Add nodes
for i in range(GRAPH_NODES):
    graph.add_node(i)

# Add edges
for i in range(GRAPH_EDGES):
    for j in range(i + 1, GRAPH_NODES):
        graph.add_edge(i, j, abs(j - i))

# Test Dijkstra's algorithm
for i in range(0,10):
    start_node = 0
    dijkstra_start_time = time.time()
    dijkstra_distances = dijkstra(graph, start_node)
    dijkstra_end_time = time.time()
    dijkstra_time = dijkstra_end_time - dijkstra_start_time
    with open("comparsion_time.log", "a") as file:
    # Write the data to the file
        text = "Dijkstra's algorithm time #" + str(i) + ":" + str(dijkstra_time)
        file.write(text)
        file.write("\n")

# Test A* algorithm
for i in range(0,10):
    goal_node = 999
    a_star_start_time = time.time()
    a_star_distance = a_star(graph, start_node, goal_node)
    a_star_end_time = time.time()
    a_star_time = a_star_end_time - a_star_start_time
    with open("comparsion_time.log", "a") as file:
    # Write the data to the file
        text = "A* algorithm time #" + str(i) + ":" + str(a_star_time)
        file.write(text)
        file.write("\n")


# # Test Node2Vec* algorithm
# nodeEmbeddings = node2Vec()
# node2vec_start_time = time.time()
# node_id = 0
# nodeEmbeddings[node_id]
# node2vec_end_time = time.time()
# node2vec_time = node2vec_end_time - node2vec_start_time
# print("Node2Vec time:", node2vec_time)