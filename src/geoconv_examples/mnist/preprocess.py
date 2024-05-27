from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.common import compute_gpc_systems

from matplotlib import pyplot as plt

import numpy as np
import scipy as sp
import trimesh
import shutil
import zipfile
import json


def create_grid(n_vertices):
    # Get mesh faces
    coordinates = np.linspace(start=0, stop=1, num=n_vertices)
    grid_vertices = np.array([(x, y) for x in coordinates for y in coordinates])
    grid_faces = sp.spatial.Delaunay(grid_vertices).simplices

    # Make vertices 3D (but keep it flat)
    grid_vertices = np.concatenate([grid_vertices, np.zeros(n_vertices ** 2).reshape(-1, 1)], axis=-1)

    return trimesh.Trimesh(vertices=grid_vertices, faces=grid_faces)


def image_to_grid(image, grid):
    grid_image = trimesh.PointCloud(grid.vertices, colors=plt.cm.binary(np.array(image).reshape((-1))))
    trimesh.Scene([grid, grid_image]).show()


def compute_bc(zipfile_path, n_radial, n_angular):
    unzipped_name = zipfile_path[:-4]
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(unzipped_name)

    with open(f"{unzipped_name}/preprocess_properties.json") as properties_file:
        properties = json.load(properties_file)
        template_radius = properties["gpc_system_radius"] * 0.75

    gpc_systems = GPCSystemGroup(object_mesh=trimesh.load_mesh(f"{unzipped_name}/normalized_mesh.stl"))
    gpc_systems.load(f"{unzipped_name}/gpc_systems")

    bc = compute_barycentric_coordinates(gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=template_radius)
    np.save(f"{unzipped_name}/BC_{n_radial}_{n_angular}_{template_radius}.npy", bc)

    print(f"Barycentric coordinates done. Zipping..")
    shutil.make_archive(base_name=unzipped_name, format="zip", root_dir=unzipped_name)
    shutil.rmtree(unzipped_name)
    print("Done.")


def preprocess(output_path, processes, n_radial, n_angular):
    # Preprocess flat grid
    grid = create_grid(n_vertices=28)
    compute_gpc_systems(grid, output_path, processes=processes)

    print(f"GPC-systems done. Zipping..")
    shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
    shutil.rmtree(output_path)
    print("Done.")

    compute_bc(f"{output_path}.zip", n_radial=n_radial, n_angular=n_angular)
