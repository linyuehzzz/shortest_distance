import random
import tensorflow as tf
import networkx as nx
import numpy as np
from timeit import default_timer



##******** Generate graph data ********##

## Number of nodes (100/1,000/10,000/100,000/1,000,000)
nodes = 1000
print('Nodes: ', nodes)
## Total degree
degree = 3
print('Degree: ', degree)

G = nx.random_regular_graph(degree,nodes)
for (u, v) in G.edges():
    G.edges[u,v]['weight'] = random.uniform(1,100)
nx.draw(G)
nx.write_weighted_edgelist(G, 'graph_n' + str(nodes) + '_d' + str(degree) + '.txt')

data = []
with open('graph_n' + str(nodes) + '_d' + str(degree) + '.txt', 'r') as f:
  lines = f.read().splitlines()
  for line in lines:
    l = line.split()
    item = [int(l[0]), int(l[1]), float(l[2])]
    data.append(item)

print(data[0])



##******** Implementation 1: list ********##
##**** Construct distance matrix ****##

def distance_matrix(graph):
  ## calculate number of nodes
  n = max([g[1] for g in graph])

  ## calculate distance matrix
  INF = float('inf')
  dist_mtx = [[INF] * n for i in range(n)]
  for g in graph:
    i = g[0] - 1
    j = g[1] - 1
    d = g[2]
    dist_mtx[i][j] = d
    dist_mtx[j][i] = d

  ## set diagonal to 0
  for i in range(n):
    dist_mtx[i][i] = 0
 
  return dist_mtx, n


## print time costs
start = default_timer()
dist_mtx, n = distance_matrix(data)
stop = default_timer()
print('List t1: ', stop - start)


##**** Calculate Hedetniemi Matrix Sum ****##

def hede_distance(matrix, n):
  INF = float('inf')
  mtx_a_t = [[INF] * n for i in range(n)]
  mtx_a_t_1 = matrix

  p = True
  while p:
    for i in range(n):
      a = mtx_a_t_1[i]
      for j in range(n):
        b = [row[j] for row in matrix]
        mtx_a_t[i][j] = min([a[k] + b[k] for k in range(n)])
    
    if mtx_a_t == mtx_a_t_1:
      p =  False
    else:
      mtx_a_t_1 = mtx_a_t   
  
  return mtx_a_t


## print time costs
start = default_timer()
mtx_a_t = hede_distance(dist_mtx, n)
stop = default_timer()
print('List t2: ', stop - start)



##******** Implementation 2: numpy ********##
##**** Construct distance matrix ****##

def distance_matrix(graph):
  ## calculate number of nodes
  n = int(np.amax(graph[:,1]))

  ## calculate distance matrix
  dist_mtx = np.full((n,n), np.inf)
  for g in graph:
    i = int(g[0]) - 1
    j = int(g[1]) - 1
    d = g[2]
    dist_mtx[i,j] = d
    dist_mtx[j,i] = d

  ## set diagonal to 0
  np.fill_diagonal(dist_mtx, 0)
 
  return dist_mtx, n


## print time costs
start = default_timer()
dist_mtx, n = distance_matrix(np.array(data))
stop = default_timer()
print('Numpy t1: ', stop - start)


##**** Calculate Hedetniemi Matrix Sum ****##

def hede_distance(matrix, n):
  mtx_a_t = np.full((n,n), np.inf)
  mtx_a_t_1 = matrix

  p = True
  while p:
    for i in range(n):
      a = mtx_a_t_1[i]
      for j in range(n):
        b = matrix[:,j]
        mtx_a_t[i,j] = np.amin([a[k] + b[k] for k in range(n)])
    
    if np.array_equal(mtx_a_t, mtx_a_t_1):
      p =  False
    else:
      mtx_a_t_1 = mtx_a_t   
  
  return mtx_a_t


## print time costs
start = default_timer()
mtx_a_t = hede_distance(dist_mtx, n)
stop = default_timer()
print('Numpy t2: ', stop - start)



##******** Implementation 3: tensorflow ********##
##**** Construct distance matrix ****##

from timeit import default_timer
import numpy as np

def distance_matrix(graph):
  ## calculate number of nodes
  n = int(np.amax(graph[:,1]))

  ## calculate distance matrix
  dist_mtx = np.full((n,n), np.inf)
  for g in graph:
    i = int(g[0]) - 1
    j = int(g[1]) - 1
    d = g[2]
    dist_mtx[i,j] = d
    dist_mtx[j,i] = d

  ## set diagonal to 0
  np.fill_diagonal(dist_mtx, 0)

  dist_mtx = tf.convert_to_tensor(dist_mtx, dtype=tf.float32)
 
  return dist_mtx, n


## print time costs
start = default_timer()
dist_mtx, n = distance_matrix(np.array(data))
stop = default_timer()
print('Tensorflow t1: ', stop - start)


##**** Calculate Hedetniemi Matrix Sum ****##

from timeit import default_timer
import tensorflow as tf
import numpy as np

def hede_distance(matrix, n):
  mtx_a_t_1 = matrix

  p = True
  while p:
    for i in tf.range(n):
      a = mtx_a_t_1[i]

      for j in tf.range(n):
        b = matrix[:,j]
        c = tf.math.reduce_min(tf.math.add(a, b))
        c = tf.fill([1], value=c)

        if tf.math.equal(j, 0):
          r = c
        else:
          r = tf.concat([r, c], 0)

      r = tf.expand_dims(r, 0)
      if tf.math.equal(i, 0):
        mtx_a_t = r
      else:
        mtx_a_t = tf.concat([mtx_a_t, r], 0)

    if tf.reduce_all(tf.math.equal(mtx_a_t_1, mtx_a_t)):
      p = False
    else:
      mtx_a_t_1 = mtx_a_t 

  return mtx_a_t

### print time costs (using gpu)
##start = default_timer()
##with tf.device('/device:GPU:0'):
##  mtx_a_t = hede_distance(dist_mtx, n)
##stop = default_timer()
##print('Tensorflow t2 using GPU: ', stop - start)

## print time costs (using cpu)
start = default_timer()
with tf.device('/cpu:0'):
  mtx_a_t = hede_distance(dist_mtx, n)
stop = default_timer()
print('Tensorflow t2 using CPU: ', stop - start)
