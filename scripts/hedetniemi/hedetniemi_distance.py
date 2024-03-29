import random
import networkx as nx
import numpy as np
import numba
import timeout_decorator
import copy
from timeit import default_timer

print(numba.config.NUMBA_NUM_THREADS)
print(numba.config.NUMBA_DEFAULT_NUM_THREADS)

##******** Read graph data ********##

## Number of nodes (100/1,000/10,000/100,000/1,000,000)
nodes = [100, 1000, 10000, 100000, 1000000]
print('Nodes: ', nodes)
## Total degree
degree = [3, 4, 5]
print('Degree: ', degree)

for i in nodes:
    for j in degree:        
        locals()['data_n' + str(i) + '_d' + str(j)] = []
        with open('graph_n' + str(i) + '_d' + str(j) + '.txt', 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                l = line.split()
                item = [int(l[0]), int(l[1]), float(l[2])]
                locals()['data_n' + str(i) + '_d' + str(j)].append(item)

        print(locals()['data_n' + str(i) + '_d' + str(j)][0])



##******** Implementation 1: list ********##
##**** Construct distance matrix ****##

@timeout_decorator.timeout(10800)
def distance_matrix_list(graph, n):
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
    dist_mtx[i][i] = 0.0
 
  return dist_mtx



##**** Calculate Hedetniemi Matrix Sum ****##

@timeout_decorator.timeout(10800)
def hede_distance_list(matrix, n):
  INF = float('inf')
  mtx_a_t = [[INF] * n for i in range(n)]
  mtx_a_t_1 = copy.deepcopy(matrix)

  for p in range(n):
    for i in range(n):
      a = mtx_a_t_1[i]      
      for j in range(n):
        b = [row[j] for row in matrix] 
        mtx_a_t[i][j] = min([a[k] + b[k] for k in range(n)])
    
    if mtx_a_t == mtx_a_t_1:
      break
    else:
      mtx_a_t_1 = copy.deepcopy(mtx_a_t)  
  
  return mtx_a_t



##******** Implementation 2: numpy ********##
##**** Construct distance matrix ****##

@timeout_decorator.timeout(10800)
def distance_matrix_np(graph, n):
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
 
  return dist_mtx



##**** Calculate Hedetniemi Matrix Sum ****##

@timeout_decorator.timeout(10800)
def hede_distance_np(matrix, n):
  mtx_a_t = np.full((n,n), np.inf)
  mtx_a_t_1 = matrix.copy()
  
  for p in range(n):
    for i in range(n):
      a = mtx_a_t_1[i]
      for j in range(n):
        b = matrix[:,j]
        mtx_a_t[i,j] = np.amin([a[k] + b[k] for k in range(n)])
    
    if np.array_equal(mtx_a_t, mtx_a_t_1):
      break
    else:
      mtx_a_t_1 = mtx_a_t.copy()   
  
  return mtx_a_t



##******** Implementation 3: numba (njit) ********##
##**** Construct distance matrix ****##

@timeout_decorator.timeout(10800)
@numba.njit
def distance_matrix_nb(graph, n):
  ## calculate distance matrix
  dist_mtx = np.full((n,n), np.inf)
  for g in numba.prange(graph.shape[0]):
    i = int(graph[g,0]) - 1
    j = int(graph[g,1]) - 1
    d = graph[g,2]
    dist_mtx[i,j] = d
    dist_mtx[j,i] = d

  ## set diagonal to 0
  np.fill_diagonal(dist_mtx, 0)
 
  return dist_mtx



##**** Calculate Hedetniemi Matrix Sum ****##

@timeout_decorator.timeout(10800)
@numba.njit
def hede_distance_nb(matrix, n):
  mtx_a_t = np.full((n,n), np.inf)
  mtx_a_t_1 = matrix.copy() 
  
  for p in numba.prange(n):
    for i in numba.prange(n):
      a = mtx_a_t_1[i]
      for j in numba.prange(n):
        b = matrix[:,j]
        mtx_a_t[i,j] = np.amin(np.array([a[k] + b[k] for k in range(n)]))
    
    if np.array_equal(mtx_a_t, mtx_a_t_1):
      break
    else:
      mtx_a_t_1 = mtx_a_t.copy()   
  
  return mtx_a_t



##******** Main ********##

with open('hedet_results.csv', 'w') as fw:
  fw.write('nodes,degree,list_t1,list_t2,np_t1,np_t2,nb_t1,nb_t2\n')
  fw.flush()
    
  for i in nodes:
    for j in degree:
      data = locals()['data_n' + str(i) + '_d' + str(j)]
      
      ## List t1
      try:
        start = default_timer()
        dist_mtx_list = distance_matrix_list(data, i)
        stop = default_timer()
        list_t1 = stop - start
      except:
        list_t1 = float('inf')
      
      ## List t2
      try:
        start = default_timer()
        mtx_a_t_list = hede_distance_list(dist_mtx_list, i)
        stop = default_timer()
        list_t2 = stop - start
        ## print shortest path matrix
        with open('hede_dist_list' + '_n' + str(i) + '_d' + str(j) + '.txt', 'w') as f:
            f.write('\n'.join(['\t'.join([str(round(cell,2)) for cell in row]) for row in mtx_a_t_list]))
      except:
        list_t2 = float('inf')
          
      ## Numpy t1
      try:
        start = default_timer()
        dist_mtx_np = distance_matrix_np(np.array(data), i)
        stop = default_timer()
        np_t1 = stop - start
      except:
        np_t1 = float('inf')
          
      ## Numpy t2
      try:
        start = default_timer()
        mtx_a_t_np = hede_distance_np(dist_mtx_np, i)
        stop = default_timer()
        np_t2 = stop - start
        ## print shortest path matrix
        with open('hede_dist_np' + '_n' + str(i) + '_d' + str(j) + '.txt', 'w') as f:
          f.write('\n'.join(['\t'.join([str(round(cell,2)) for cell in row]) for row in mtx_a_t_np.tolist()]))                
      except:
        np_t2 = float('inf')
          
      ## Numba (njit) t1
      try:
        start = default_timer()
        dist_mtx_nb = distance_matrix_nb(np.array(data), i)
        stop = default_timer()
        nb_t1 = stop - start
      except:
        nb_t1 = float('inf')
      
      ## Numba (njit) t2
      try:
        start = default_timer()
        mtx_a_t_nb = hede_distance_nb(dist_mtx_nb, i)
        stop = default_timer()
        nb_t2 = stop - start
        ## print shortest path matrix
        with open('hede_dist_nb' + '_n' + str(i) + '_d' + str(j) + '.txt', 'w') as f:
          f.write('\n'.join(['\t'.join([str(round(cell,2)) for cell in row]) for row in mtx_a_t_nb.tolist()]))                
      except:
        nb_t2 = float('inf')
          
      fw.write(str(i) + ',' + str(j) + ',' + str(list_t1) + ',' + str(list_t2) + ','
                + str(np_t1) + ',' + str(np_t2) + ',' + str(nb_t1) + ',' + str(nb_t2) + '\n')

      fw.flush()
fw.close()
