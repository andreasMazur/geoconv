from geoconv.utils.misc import get_faces_of_edge

from io import BytesIO

import zipfile
import trimesh
import numpy as np
import os
import subprocess
import random


def remove_non_manifold_edges(mesh):
    """Removes non-manifold edges by removing all their faces.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The manifold mesh.

    Returns
    -------
    trimesh.Trimesh:
        The mesh without non-manifold edges.
    """
    # Check if non-manifold edges exist
    non_manifold_edges = np.asarray(mesh.as_open3d.get_non_manifold_edges())
    if non_manifold_edges.shape[0] > 0:
        # Compute mask that removes non-manifold edges and all their faces
        face_mask = np.full(mesh.faces.shape[0], True)
        for edge in non_manifold_edges:
            sorted_edge, edge_faces = get_faces_of_edge(edge, mesh)
            for edge_f in edge_faces:
                update_mask = np.logical_not((edge_f == mesh.faces).all(axis=-1))
                face_mask = np.logical_and(face_mask, update_mask)
        # Remove non-manifold edges and faces with mask
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[face_mask])
    return mesh


def down_sample_mesh(mesh, target_number_of_triangles):
    """Down-samples the mesh to (roughly) the given target number of triangles.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The mesh to be down-sampled.
    target_number_of_triangles: int
        The target amount of triangles. Cannot be ensured.

    Returns
    -------
    trimesh.Trimesh:
        The down-sampled mesh.
    """
    mesh = mesh.as_open3d.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)
    trimesh.repair.fill_holes(mesh)
    return mesh


def zip_file_generator(zipfile_path,
                       file_type,
                       manifold_plus_executable=None,
                       down_sample=None,
                       return_filename=False,
                       min_vertices=100,
                       timeout_in_sec=20,
                       mp_depth=8,
                       shape_path_contains=None,
                       epsilon=0.25):
    """Loads shapes from a given zip-file and removes non-manifold edges.

    Parameters
    ----------
    zipfile_path: str
        The path to the zip-file.
    file_type: str
        The file type of the CAD models.
    manifold_plus_executable: str
        The path to the manifold+ algorithm.
    down_sample: int
        The target amount of triangles the mesh shall be down-sampled to.
    return_filename: bool
        Whether to return the filename of the shape within the zip-file.
    min_vertices: int
        The minimal amount of vertices a shape shall return.
    timeout_in_sec: int
        The amount of seconds to wait before killing the manifold+ subprocess and continue with the next shape.
    mp_depth: int
        Depth for manifold+-algorithm.
    shape_path_contains: list
        A list of strings that is contained in the shape-path within the zip-file. If none of the contained strings are
        within the shape-path, then the shape is skipped.
    epsilon: float
        Percentage value that describes how far a down-sampled shape can deviate from the target amount of faces
        given by the parameter 'down_sample'. If 'down_sample' is 'None' this value is not used.

    Returns
    -------
    trimesh.Trimesh:
        A manifold-mesh.
    """
    # Load the zip-file
    zip_file = zipfile.ZipFile(zipfile_path, "r")
    zip_content = [fn for fn in zip_file.namelist() if fn[-3:] == file_type]
    zip_content.sort()

    for shape_path in zip_content:
        # Skip irrelevant shapes
        if shape_path_contains is not None:
            if np.all([substring not in shape_path for substring in shape_path_contains]):
                continue

        # Load shape
        shape = trimesh.load_mesh(BytesIO(zip_file.read(shape_path)), file_type=file_type)

        # Concat meshes if a scene was loaded
        if type(shape) == trimesh.scene.Scene:
            shape = trimesh.util.concatenate([y for y in shape.geometry.values()])

        if manifold_plus_executable is not None:
            # Create temporary file for manifold+ algorithm
            in_file = f"./in_shape_temp.obj"
            shape.export(in_file)

            # Create output file
            out_file = f"./out_shape_temp.obj"

            # Manifold plus algorithm
            if np.asarray(shape.as_open3d.get_non_manifold_edges()).shape[0] > 0:
                try:
                    subprocess.run(
                        [manifold_plus_executable, "--input", in_file, "--output", out_file, "--depth", f"{mp_depth}"],
                        timeout=timeout_in_sec
                    )
                except subprocess.TimeoutExpired:
                    print(
                        f"*** {shape_path} took more than {timeout_in_sec} to be processed by the manifold+ algorithm. "
                        f"Skipping to next shape."
                    )
                    continue
                shape = trimesh.load_mesh(out_file)
                # Remove temp files
                os.remove(out_file)

            # Remove temp files
            os.remove(in_file)

        # Simplify shape
        if down_sample is not None and shape.faces.shape[0] > down_sample:
            shape = down_sample_mesh(shape, down_sample)
            # Check result and skip if it is too far from target amount of faces
            if shape.faces.shape[0] > down_sample + down_sample * epsilon:
                print(f"*** {shape_path} couldn't be down-sampled enough to {down_sample}.")
                continue

        # Remove non-manifold edges
        shape = remove_non_manifold_edges(shape)

        if shape.vertices.shape[0] > min_vertices and shape.faces.shape[0] > 0:
            if return_filename:
                yield shape, shape_path
            else:
                yield shape
        else:
            print(f"{shape_path} has less then {min_vertices} vertices. Skipping to next shape..")


def preprocessed_shape_generator(zipfile_path,
                                 filter_list,
                                 sorting_key=None,
                                 shuffle_seed=None,
                                 split=None):
    """Loads all shapes within a preprocessed dataset and filters within each shape-directory for files.

    This function sorts alphanumerically after the shape-directory name.

    Parameters
    ----------
    zipfile_path: str
        The path to the preprocessed dataset.
    filter_list: list
        A list of substrings to filter for in each shape-directory.
    sorting_key: callable
        A function that takes a single file-path as an argument and returns its part after which it should be sorted.
        If 'None' is given, then sorting is performed with respect to the directory name of a shape, i.e., the shapes
        name.
    shuffle_seed: int
        Whether to randomly shuffle the data with the given seed. If no seed is given, no shuffling will be performed.
    split: list
        List of integers which are yielded from the list of all shapes.

    Returns
    -------
    list:
        The loaded files and their corresponding filenames.
    """
    # Load the zip-file
    zip_file = np.load(zipfile_path)

    # Get and sort all shape directories from preprocessed shapes
    preprocessed_shapes = ["/".join(fn.split("/")[:-1]) for fn in zip_file.files if "preprocess_properties.json" in fn]

    # Sort shapes
    if sorting_key is None:
        def sorting_key(file_name):
            return file_name.split("/")[-1]
    preprocessed_shapes.sort(key=sorting_key)

    # Shuffle shapes
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(preprocessed_shapes)

    # Filter for given split
    preprocessed_shapes = np.array(preprocessed_shapes)
    if split is not None:
        preprocessed_shapes = preprocessed_shapes[split]

    # Iterate over dataset shapes
    for preprocessed_shape_dir in preprocessed_shapes:

        # Iterate over shape's data and collect with filters
        preprocessed_shape_dir = [x for x in zip_file.files if preprocessed_shape_dir in x and "gpc_systems" not in x]

        # Seek for file-names that contain a given filter string as a sub-string
        shape_files = []
        for filter_str in filter_list:
            for file_name in preprocessed_shape_dir:
                if filter_str in file_name:
                    shape_files.append(file_name)

        yield [(zip_file[shape_file], shape_file) for shape_file in shape_files]


def barycentric_coordinates_generator(zipfile_path,
                                      n_radial,
                                      n_angular,
                                      template_radius,
                                      return_filename=False):
    """Loads barycentric coordinates from a preprocessed dataset

    Parameters
    ----------
    zipfile_path: str
        Path to preprocessed dataset.
    n_radial: int
        Amount of radial coordinates contained in the barycentric coordinates that shall be loaded.
    n_angular: int
        Amount of angular coordinates contained in the barycentric coordinates that shall be loaded.
    template_radius: float
        The template radius considered during barycentric coordinates computation.
    return_filename: bool
        Whether to return the filename of the shape within the zip-file.

    Returns
    -------
    np.ndarray:
        Barycentric coordinates.
    """
    psg = preprocessed_shape_generator(zipfile_path, filter_list=[f"BC_{n_radial}_{n_angular}_{template_radius}"])
    for filtered_files in psg:
        if return_filename:
            yield filtered_files[0][0], filtered_files[0][1]
        else:
            yield filtered_files[0][0]
