import numpy as np
import trimesh
import os


def raw_data_generator(path, return_file_name=False):
    """Loads the manually labeled data."""
    for file_name in os.listdir(f"{path}/out_data"):
        d = np.load(f'{path}/out_data/{file_name}')
        if return_file_name:
            yield trimesh.Trimesh(vertices=d["verts"], faces=d["faces"]), d["labels"], file_name
        else:
            yield trimesh.Trimesh(vertices=d["verts"], faces=d["faces"]), d["labels"]
