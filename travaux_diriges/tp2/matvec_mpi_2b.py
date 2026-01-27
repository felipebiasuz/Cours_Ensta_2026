from mpi4py import MPI
import numpy as np

t0 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

N = 120
Nloc = N // nbp

# Initialisation
u_loc = np.array([i + 1. for i in range(N)])

A_loc = np.zeros((Nloc, N))

i_start = rank * Nloc
i_end = (rank + 1) * Nloc
for i_loc, i in enumerate(range(i_start, i_end)):
    for j in range(N):
        A_loc[i_loc, j] = (i + j) % N + 1

# Synchronisation AVANT le chrono
comm.Barrier()

# Calcul local
v_loc = A_loc.dot(u_loc)

# Réduction
v = None
if rank == 0:
    v = np.zeros(N)  # vetor resultado final
comm.Gather(v_loc, v, root=0)

# Synchronisation APRÈS
comm.Barrier()

t1 = MPI.Wtime()
elapsed = t1 - t0

if rank == 0:
    print("v =", v)
    print(f"Temps d'exécution parallel = {elapsed:.6f} s")