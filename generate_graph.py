import random
import networkx as nx

## Number of nodes (100/1,000/10,000/100,000/1,000,000)
nodes = [100, 1000, 10000, 100000, 1000000]
print('Nodes: ', nodes)
## Total degree
degree = [3, 4, 5]
print('Degree: ', degree)

for i in nodes:
    for j in degree:
        G = nx.random_regular_graph(j, i)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = random.uniform(1,100)
        nx.write_weighted_edgelist(G, 'graph_n' + str(i) + '_d' + str(j) + '.txt')
