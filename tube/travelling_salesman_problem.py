from __future__ import annotations

import numpy as np
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.exact import solve_tsp_branch_and_bound
from python_tsp.exact import solve_tsp_brute_force
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search
from python_tsp.utils import compute_permutation_distance

sources = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
destinations = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

distance_matrix = euclidean_distance_matrix(sources, destinations)
distance_matrix[np.diag_indices_from(distance_matrix)] = 1000
print(distance_matrix)
permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
print(permutation, distance)

xopt, fopt = solve_tsp_brute_force(distance_matrix)
print(xopt, fopt)


xopt, fopt = solve_tsp_branch_and_bound(distance_matrix)
print(xopt, fopt)


xopt, fopt = solve_tsp_local_search(distance_matrix)
print(xopt, fopt)


print(compute_permutation_distance(distance_matrix, permutation=xopt))
print(sources[xopt])
