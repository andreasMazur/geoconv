from geoconv.utils.misc import get_faces_of_edge

from io import BytesIO

import zipfile
import trimesh
import numpy as np
import os


def remove_nme(mesh):
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
                       min_vertices=100):
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

    Returns
    -------
    trimesh.Trimesh:
        A manifold-mesh.
    """
    zip_file = zipfile.ZipFile(zipfile_path, "r")
    zip_content = [fn for fn in zip_file.namelist() if fn[-3:] == file_type]
    zip_content.sort()

    for shape_path in zip_content:
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
                os.system(f"{manifold_plus_executable} --input {in_file} --output {out_file} --depth 8")
                shape = trimesh.load_mesh(out_file)
                # Remove temp files
                os.remove(out_file)

            # Remove temp files
            os.remove(in_file)

        if down_sample is not None and shape.faces.shape[0] > down_sample:
            shape = down_sample_mesh(shape, down_sample)

        # Remove non-manifold meshes
        shape = remove_nme(shape)

        if shape.vertices.shape[0] > min_vertices:
            if return_filename:
                yield shape, shape_path
            else:
                yield shape
        else:
            print(f"{shape_path} has less then {min_vertices} vertices. Skipping to next shape..")
