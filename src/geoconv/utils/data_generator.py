from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import get_faces_of_edge, repair_mesh, normalize_mesh
from geoconv.utils.visualization import draw_gpc_on_mesh, draw_gpc_triangles

from io import BytesIO
from tqdm import tqdm

import zipfile
import trimesh
import numpy as np
import os
import subprocess
import random
import json
import re
import math


def remove_nme(mesh):
    """Removes non-manifold edges by removing all their faces.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The triangle mesh.

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
    """Tries to down-samples the mesh to the given target number of triangles. Target number is not guaranteed.

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
                       target_amount_faces=None,
                       return_filename=False,
                       min_vertices=100,
                       timeout_in_sec=20,
                       mp_depth=8,
                       shape_path_contains=None,
                       epsilon=0.25,
                       remove_non_manifold_edges=True,
                       normalize=False,
                       repair_shapes=True):
    """Loads shapes from a given zip-file and removes non-manifold edges.

    Parameters
    ----------
    zipfile_path: str
        The path to the zip-file.
    file_type: str
        The file type of the CAD models.
    manifold_plus_executable: str
        The path to the manifold+ algorithm.
    target_amount_faces: int
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
    remove_non_manifold_edges: bool
        Whether to remove non-manifold edges.
    normalize: bool
        Whether to normalize the mesh.
    repair_shapes: bool
        Whether to merge close vertices and remove degenerate faces.

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
        if target_amount_faces is not None and shape.faces.shape[0] > target_amount_faces:
            shape = down_sample_mesh(shape, target_amount_faces)
            # Check result and skip if it is too far from target amount of faces
            if shape.faces.shape[0] > target_amount_faces + target_amount_faces * epsilon:
                print(f"*** {shape_path} couldn't be down-sampled close enough to {target_amount_faces}.")
                continue

        # Remove non-manifold edges
        if remove_non_manifold_edges:
            shape = remove_nme(shape)

        # Merge vertices
        if repair_shapes:
            shape = repair_mesh(shape)

        # Normalize shape
        if normalize:
            try:
                shape, geodesic_diameter = normalize_mesh(shape)
            except RuntimeError:
                print(f"{shape_path} crashed during normalization.")
                continue

        if shape.vertices.shape[0] > min_vertices and shape.faces.shape[0] > 0:
            if return_filename:
                yield shape, shape_path
            else:
                yield shape
        else:
            print(f"{shape_path} has less then {min_vertices} vertices. Skipping to next shape..")


def preprocessed_shape_generator(zipfile_path,
                                 filter_list,
                                 batch_size,
                                 sorting_key=None,
                                 generator_info="",
                                 shuffle_seed=None,
                                 directive=None):
    """Loads all shapes within a preprocessed dataset and filters within each shape-directory for files.

    This function sorts alphanumerically after the shape-directory name.
    It is mandatory that the folder that contain shape-files contain a 'preprocess_properties.json'-file.
    Directories that do not contain it, will be discarded.

    Parameters
    ----------
    zipfile_path: str
        The path to the preprocessed dataset.
    filter_list: list
        A list of substrings to filter for in each shape-directory.
    sorting_key: callable | None
        A function that takes a single file-path as an argument and returns its part after which it should be sorted.
        If 'None' is given, then sorting is performed with respect to the directory name of a shape, i.e., the shapes
        name.
    shuffle_seed: int | None
        Whether to randomly shuffle the data with the given seed. If no seed is given, no shuffling will be performed.
    batch_size: int
        How many shapes to return per iteration.
    generator_info: str
        A file where the generator can store data that was computed during generator preparation. If file already
        exists, the generator will load the information stored there instead of conducting the preparation process
        again.
    directive: function | None
        A function that receives a file-dictionary, does something with it, and returns a modified file-dictionary.
        The file-dictionary contains all files that were allowed by the user-provided filters.

    Returns
    -------
    list:
        The loaded files and their corresponding filenames.
    """
    # Load raw zip
    zip_file = np.load(zipfile_path)

    # Load shapes (shapes have to be in a folder with a 'preprocess_properties.json'-file)
    preprocessed_shapes = ["/".join(fn.split("/")[:-1]) for fn in zip_file.files if "preprocess_properties.json" in fn]

    # Sort shape directories
    if sorting_key is None:
        def sorting_key(file_name):
            return file_name.split("/")[-1]
    preprocessed_shapes.sort(key=sorting_key)

    # Remember zip-file content
    if os.path.exists(generator_info):
        # Check whether generator info is available
        with open(generator_info, "r") as gen_info_fd:
            sfp_dict = json.load(gen_info_fd)
        shape_file_paths = [psf for psf in sfp_dict.values()]
    else:
        def check_path(path):
            for f in filter_list:
                if re.search(f, path) is not None:
                    return True
            return False

        # Iterate over shape's data and collect with filters
        shape_file_paths = []
        for shape_dir_content in tqdm(preprocessed_shapes, postfix="Preparing generator.."):
            shape_file_paths.append([x for x in zip_file.files if shape_dir_content in x and check_path(x)])
        shape_file_paths = list(filter(lambda x: x != [], shape_file_paths))

        # Apply user-defined directive on file-dictionary
        sfp_dict = {}
        for sfp in shape_file_paths:
            sfp_dict["/".join(sfp[0].split("/")[:-1])] = sfp
        if directive is not None:
            sfp_dict = directive(sfp_dict)
            shape_file_paths = [psf for psf in sfp_dict.values()]

        # Shuffle shapes
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(shape_file_paths)

        # Store generator's preparation results
        if len(generator_info) > 0:
            with open(generator_info, "w") as gen_info_fd:
                json.dump(sfp_dict, gen_info_fd, indent=4)

    # Batching
    n_batches = math.ceil(len(shape_file_paths) / batch_size)
    shape_file_paths = [shape_file_paths[i * batch_size:(i * batch_size) + batch_size] for i in range(n_batches)]

    # Return batch of files and their corresponding file names
    for batch in shape_file_paths:
        yield [(zip_file[shape_file], shape_file) for directory in batch for shape_file in directory]


def preprocessed_properties_generator(zipfile_path, return_filename=False, sorting_key=None):
    """Loads all shape properties files from preprocessed dataset. First yielded file describes dataset properties.

    Parameters
    ----------
    zipfile_path: str
        The path to the preprocessed dataset.
    return_filename: bool
        Whether to return the file-path of the properties file within the zip-file.
    sorting_key: callable
        A function that takes a single file-path as an argument and returns its part after which it should be sorted.
        If 'None' is given, then sorting is performed with respect to the directory name of a shape, i.e., the shapes
        name.

    Returns
    -------
    dict, str:
        A properties dictionary and, if wanted, the path of the properties file within the zip-file.
    """
    # Load the zip-file
    print(f"Loading properties from zip-file.. ({zipfile_path})")
    zip_file = np.load(zipfile_path)
    print("Done.")

    # Get and sort all shape directories from preprocessed shapes
    pr_str = "preprocess_properties.json"
    properties_files = [f"{'/'.join(fn.split('/')[:-1])}/{pr_str}" for fn in zip_file.files if pr_str in fn]
    properties_files = ["dataset_properties.json"] + properties_files

    # Sort properties files
    if sorting_key is None:
        def sorting_key(file_name):
            return file_name.split("/")[-1]
    properties_files.sort(key=sorting_key)

    # Yield properties and, if wanted, properties path
    for properties_path in properties_files:
        if return_filename:
            yield json.load(BytesIO(zip_file[properties_path])), properties_path
        else:
            yield json.load(BytesIO(zip_file[properties_path]))


def barycentric_coordinates_generator(zipfile_path,
                                      n_radial,
                                      n_angular,
                                      template_radius,
                                      batch_size,
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
    batch_size: int
        The batch size.
    return_filename: bool
        Whether to return the filename of the shape within the zip-file.

    Returns
    -------
    np.ndarray:
        Barycentric coordinates.
    """
    psg = preprocessed_shape_generator(
        zipfile_path, batch_size=batch_size, filter_list=[f"BC_{n_radial}_{n_angular}_{template_radius}"]
    )
    for filtered_files in psg:
        if return_filename:
            yield filtered_files[0][0], filtered_files[0][1]
        else:
            yield filtered_files[0][0]


def read_template_configurations(zipfile_path):
    """Reads the template configurations stored within a preprocessed dataset.

    Parameters
    ----------
    zipfile_path: str
        The path to the preprocessed dataset.

    Returns
    -------
    list:
        A list of tuples of the form (n_radial, n_angular, template_radius). These configurations have been
        found in the given zipfile.
    """
    # Load zip-file
    zip_file = np.load(zipfile_path)

    # Load template configuration dictionary
    template_configurations = json.load(BytesIO(zip_file["dataset_properties.json"]))["template_configurations"]

    # Convert to list and return
    as_list = []
    for temp_conf in template_configurations.values():
        as_list.append((temp_conf["n_radial"], temp_conf["n_angular"], temp_conf["template_radius"]))
    return as_list


def inspect_gpc_systems(zipfile_path, shuffle=True, show_all_gpc_systems=False):
    """Inspect the GPC-systems within a preprocessed dataset.

    Parameters
    ----------
    zipfile_path: str
        The path to the preprocessed dataset.
    shuffle: bool
        Whether to shuffle the dataset
    show_all_gpc_systems:
        Whether to only show all GPC-systems. If set to 'False', three random GPC-systems will be displayed.
    """
    psg = preprocessed_shape_generator(
        zipfile_path=zipfile_path,
        filter_list=["stl", "gpc_systems/.+"],
        shuffle_seed=42 if shuffle else None,
        batch_size=False
    )
    for content_list in psg:
        # Load GPC-systems and properties from bytes
        shape_dict = {}
        shape, content_path = None, None
        for (content, content_path) in content_list:
            if content_path[-3:] == "stl":
                shape = trimesh.load_mesh(BytesIO(content), file_type="stl")
            elif content_path[-4:] == "json":
                shape_dict[content_path.split("/")[-1][:-5]] = json.load(BytesIO(content))
            else:
                shape_dict[content_path.split("/")[-1][:-4]] = np.load(BytesIO(content))

        assert shape is not None and content_path is not None, "No shape found."

        # Load GPC-systems from dict
        gpc_systems = GPCSystemGroup(object_mesh=shape)
        gpc_systems.load(from_dict=shape_dict)
        indices = list(range(len(shape_dict["properties"].keys())))

        # Get 3 random GPC-systems
        if not show_all_gpc_systems:
            random.shuffle(indices)
            indices = indices[:3]

        # Visualize GPC-systems
        print(f"Currently observing: {'/'.join(content_path.split('/')[:-1])}")
        for gpc_system_idx in indices:
            draw_gpc_triangles(
                gpc_systems.object_mesh_gpc_systems[gpc_system_idx],
                plot=True,
                title=f"GPC-system",
            )
            draw_gpc_on_mesh(
                gpc_system_idx,
                gpc_systems.object_mesh_gpc_systems[gpc_system_idx].radial_coordinates,
                gpc_systems.object_mesh_gpc_systems[gpc_system_idx].angular_coordinates,
                shape
            )
