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
  # define an array in the shared memory
  if dimBlock[0] == 4:
    s_a = cuda.shared.array(shape=(4, 4), dtype=float32)
    s_b = cuda.shared.array(shape=(4, 4), dtype=float32)
  elif dimBlock[0] == 8:
    s_a = cuda.shared.array(shape=(8, 8), dtype=float32)
    s_b = cuda.shared.array(shape=(8, 8), dtype=float32)
  elif dimBlock[0] == 16:
    s_a = cuda.shared.array(shape=(16, 16), dtype=float32)
    s_b = cuda.shared.array(shape=(16, 16), dtype=float32)
  elif dimBlock[0] == 32:
    s_a = cuda.shared.array(shape=(32, 32), dtype=float32)
    s_b = cuda.shared.array(shape=(32, 32), dtype=float32)
  
  x, y = cuda.grid(2)

  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y

  bpg = cuda.gridDim.x
  tpb = cuda.blockDim.x

  if x >= n and y >= n:
    return
  
  # calculate matrix t
  summ = np.inf
  for i in range(bpg):
    # preload data into shared memory
    s_a[tx, ty] = mtx_a_t_1[x, ty + i * tpb]
    s_b[tx, ty] = matrix[tx + i * tpb, y]
    cuda.syncthreads()
    for j in range(tpb):
      summ = min(summ, s_a[tx, j] + s_b[j, ty])
    cuda.syncthreads()
  mtx_a_t[x,y] = summ

  # compare matrix t and matrix t-1
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



##******** Compile ********##
d = [[1, 2, 30], [1, 4, 30], [1, 9, 40],
        [2, 3, 25], [2, 4, 40], [3, 4, 50],
        [4, 5, 30], [4, 6, 20], [5, 7, 25],
        [6, 7, 20], [6, 9, 20], [7, 8, 25],
        [8, 9, 20]]
n = 9

dimGrid, dimBlock = get_cuda_execution_config(n, 4)
dist_mtx = distance_matrix(np.array(d), n)
mtx_a_t = hede_distance(dist_mtx, n)



##******** Main ********##

with open('hedet_results_nb_cuda_sm.csv', 'w') as fw:
  fw.write('bpg,tpb,nodes,degree,nb_cuda_sm_t1,nb_cuda_sm_t2\n')
  fw.flush()

  for k in NUM_THREADS:
    for i in nodes:
      dimGrid, dimBlock = get_cuda_execution_config(i, k)
      if dimGrid[0] > 65535 or i % k != 0:
        continue
      
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
          with open('hedet_dist_nb_cuda_sm' + '_n' + str(i) + '_d' + str(j) + '.txt', 'w') as f:
            f.write('\n'.join(['\t'.join([str(round(cell,2)) for cell in row]) for row in mtx_a_t_hedet.tolist()]))
        except:
          cuda_hedet_t2 = float('inf')
            
        fw.write(str(dimGrid) + ',' + str(dimBlock) + ',' + str(i) + ',' + str(j) + ',' + str(cuda_t1) + ',' + str(cuda_hedet_t2) + '\n')

        fw.flush()
