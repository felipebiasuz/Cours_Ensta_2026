# Produit matrice-vecteur v = A.u
import numpy as np
import time

t0 = time.perf_counter()

# Dimension du problème (peut-être changé)
dim = 120
# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])

# Produit matrice-vecteur
v = A.dot(u)

t1 = time.perf_counter()
elapsed = t1 - t0

print(f"A = {A}")
print(f"u = {u}")
print(f"v = {v}")

print(f"Temps d'exécution séquentiel = {elapsed:.6f} s")