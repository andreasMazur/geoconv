from geoconv.preprocessing.dgpc.dgpc_solver import DgpcSolver
from tqdm import tqdm

import numpy as np


def pickable_dgpc(vertex_indices, vertices, faces, u_max):
    solver = DgpcSolver(V=vertices, F=faces, u_max=u_max)
    gpc_systems = []
    for source_point in tqdm(vertex_indices):
        gpc_systems.append(np.stack(solver.compute_dgpc(source_point), axis=-1))
    gpc_systems = np.array(gpc_systems)
    # gpc_systems = np.array([np.stack(solver.compute_dgpc(source_point), axis=-1) for source_point in vertex_indices])
    return gpc_systems
