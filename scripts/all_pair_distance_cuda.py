from timeit import default_timer
from numba import cuda, njit, float32
import numpy as np
import math
import timeout_decorator


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



##******** Configure CUDA ********##

# number of threads per block: 4*4, 8*8, 16*16, 32*32
NUM_THREADS = [4, 8, 16, 32]

def get_cuda_execution_config(n, tpb):
  dimBlock = (tpb, tpb)
  dimGrid = (math.ceil(n / tpb), math.ceil(n / tpb))

  return dimGrid, dimBlock



##******** Construct distance matrix ********##

@cuda.jit
def graph2dist(graph, dist_mtx, n):
  stride = cuda.gridDim.x * cuda.blockDim.x

  ## initialize distance matrix
  x, y = cuda.grid(2)
  if x < n and y < n:
    dist_mtx[x,y] = np.inf

  ## calculate distance matrix
  for i in range(x, graph.shape[0], stride):
    a = int(graph[i,0]) - 1
    b = int(graph[i,1]) - 1
    d = graph[i,2]
    dist_mtx[a,b] = d
    dist_mtx[b,a] = d
  
  ## set diagonal to 0
  if x < n:
    dist_mtx[x,x] = 0.0


@timeout_decorator.timeout(10800)
def distance_matrix(graph, n):
  ## copy data to device
  graph_device = cuda.to_device(graph)
  dist_mtx_device = cuda.device_array(shape=(n,n))

  ## calculate distance matrix
  graph2dist[dimGrid, dimBlock](graph_device, dist_mtx_device, n)
  
  ## copy data to host
  dist_mtx_host = dist_mtx_device.copy_to_host()
 
  return dist_mtx_host



##**** Hedetniemi distance ****##

@cuda.jit
def init_mtx(matrix, mtx_a_t_1, mtx_a_t, n):
  # initialize distance matrix
  x, y = cuda.grid(2)
  if x < n and y < n:
    mtx_a_t[x,y] = np.inf
    mtx_a_t_1[x,y] = matrix[x,y]


@cuda.jit
def all_pair_hedet(matrix, mtx_a_t_1, mtx_a_t, n, p):
  x, y = cuda.grid(2)
  if x < n and y < n:
    summ = np.inf
    for k in range(n):
      summ = min(summ, mtx_a_t_1[x, k] + matrix[k, y])
    mtx_a_t[x,y] = summ

    if mtx_a_t_1[x,y] != mtx_a_t[x,y]:    
      mtx_a_t_1[x,y] = mtx_a_t[x,y]
      p[0] = False


@timeout_decorator.timeout(10800)
def hede_distance(matrix, n):
  ## copy data to device
  matrix_device = cuda.to_device(matrix)
  mtx_a_t_1_device = cuda.device_array(shape=(n,n))
  mtx_a_t_device = cuda.device_array(shape=(n,n))

  ## initialize hedetniemi distance
  init_mtx[dimGrid, dimBlock](matrix_device, mtx_a_t_1_device, mtx_a_t_device, n)

  ## calculate hedetniemi distance
  for k in range(n):
    p = cuda.to_device([True])
    all_pair_hedet[dimGrid, dimBlock](matrix_device, mtx_a_t_1_device, mtx_a_t_device, n, p)
    if p[0] == True:
      break
  
  ## copy data to host
  mtx_a_t_host = mtx_a_t_device.copy_to_host()
 
  return mtx_a_t_host



##******** Floyd–Warshall distance ********##

@cuda.jit
def all_pair_floyd(matrix, k, n):
  x, y = cuda.grid(2)
  if x < n and y < n:
    matrix[x,y] = min(matrix[x,y], matrix[x,k] + matrix[k,y])


@timeout_decorator.timeout(10800)
def floyd_distance(matrix, n):
  ## copy data to device
  matrix_device = cuda.to_device(matrix)

  ## calculate hedetniemi distance
  for k in range(n):
    all_pair_floyd[dimGrid, dimBlock](matrix_device, k, n)
  
  ## copy data to host
  matrix_host = matrix_device.copy_to_host()
 
  return matrix_host



##******** Main ********##

with open('all_pair_cuda_results.csv', 'w') as fw:
  fw.write('bpg,tpb,nodes,degree,t1,hedet_t2,floyd_t2\n')
  fw.flush()

  for k in NUM_THREADS:
    for i in nodes:
      dimGrid, dimBlock = get_cuda_execution_config(i, k)
      
      for j in degree:
        data = locals()['data_n' + str(i) + '_d' + str(j)]
        
        ## distance matrix
        try:
          start = default_timer()
          dist_mtx = distance_matrix(np.array(data), i)
          stop = default_timer()
          cuda_t1 = stop - start
        except:
          cuda_t1 = float('inf')
        
        ## hedetniemi distance
        try:
          start = default_timer()
          mtx_a_t_hedet = hede_distance(dist_mtx, i)
          stop = default_timer()
          cuda_hedet_t2 = stop - start
          ## print shortest path matrix
          with open('hedet_dist_nb_cuda' + '_n' + str(i) + '_d' + str(j) + '.txt', 'w') as f:
            f.write('\n'.join(['\t'.join([str(round(cell,2)) for cell in row]) for row in mtx_a_t_hedet.tolist()]))
        except:
          cuda_hedet_t2 = float('inf')
            
        ## floyd distance
        try:
          start = default_timer()
          mtx_a_t_floyd = floyd_distance(dist_mtx, i)
          stop = default_timer()
          cuda_floyd_t2 = stop - start
          ## print shortest path matrix
          with open('floyd_dist_nb_cuda' + '_n' + str(i) + '_d' + str(j) + '.txt', 'w') as f:
            f.write('\n'.join(['\t'.join([str(round(cell,2)) for cell in row]) for row in mtx_a_t_floyd.tolist()]))                
        except:
          cuda_floyd_t2 = float('inf')       
            
        fw.write(str(dimGrid) + ',' + str(dimBlock) + ',' + str(i) + ',' + str(j) + ',' + str(cuda_t1) + ',' + str(cuda_hedet_t2) + ','
                  + str(cuda_floyd_t2) + '\n')

        fw.flush()
