from mpi4py import MPI
import numpy as np

t0 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

N = 120
Nloc = N // nbp

# Initialisation
j_start = rank * Nloc
j_end = (rank + 1) * Nloc
u = np.array([i + 1. for i in range(j_start, j_end)])

A_loc = np.zeros((N, Nloc))

for i in range(N):
    for j_loc, j in enumerate(range(j_start, j_end)):
        A_loc[i, j_loc] = (i + j) % N + 1

# Synchronisation AVANT le chrono
comm.Barrier()

# Calcul local
v_loc = A_loc.dot(u)

# Réduction
v = np.zeros(N)
comm.Allreduce(v_loc, v, op=MPI.SUM)

# Synchronisation APRÈS
comm.Barrier()

t1 = MPI.Wtime()
elapsed = t1 - t0

if rank == 0:
    print("v =", v)
    print(f"Temps d'exécution parallel = {elapsed:.6f} s")