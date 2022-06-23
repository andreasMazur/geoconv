import numpy as np
from scipy import sparse

import os


class MeshData:

    def __init__(self, path):
        self.path = path
        self.directory = os.listdir(path)
        self.directory.sort()
        self.n_meshes = len(self.directory)
        self.current_idx = None

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):

        if self.current_idx >= self.n_meshes:
            raise StopIteration

        # TODO
        mesh_folder = self.directory[self.current_idx]
        mesh_folder = os.listdir(f"{self.path}/{mesh_folder}")
        for matrix in mesh_folder:
            matrix = sparse.load_npz(matrix)
            matrix = matrix.to_array()

        self.current_idx += 1
