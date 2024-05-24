from geoconv.utils.misc import get_faces_of_edge

from io import BytesIO

import numpy as np
import trimesh
import os
import zipfile


def down_sample_mesh(mesh, target_number_of_triangles):
    """Down-samples the mesh."""
    mesh = mesh.as_open3d.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)
    trimesh.repair.fill_holes(mesh)
    return mesh


def remove_nme(mesh):
    """Removes non-manifold edges by removing all their faces."""
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


def up_shapenet_generator(shapenet_path,
                          return_filename=False,
                          remove_non_manifold_edges=False,
                          down_sample=None,
                          synset_ids=None):
    """Yields unprocessed shapenet shapes directly from the zip-files."""
    # Check for file ending
    synset_ids = [synset_id if synset_id[-3:] == "zip" else f"{synset_id}.zip" for synset_id in synset_ids]

    # Synset (WordNet): Set of synonyms
    if synset_ids is None:
        synset_ids = os.listdir(shapenet_path)

    for synset_id in synset_ids:
        synset_ids_zip = zipfile.ZipFile(f"{shapenet_path}/{synset_id}", "r")
        zip_content = [fn for fn in synset_ids_zip.namelist() if fn[-3:] == "obj"]
        zip_content.sort()
        for shape_path in zip_content:
            # Load shape
            mesh = trimesh.load_mesh(BytesIO(synset_ids_zip.read(shape_path)), file_type="obj")

            # Concat meshes if a scene was loaded
            if type(mesh) == trimesh.scene.Scene:
                mesh = trimesh.util.concatenate([y for y in mesh.geometry.values()])

            # Down-sample mesh
            if down_sample is not None and mesh.faces.shape[0] > down_sample:
                mesh = down_sample_mesh(mesh, down_sample)

            # Remove non-manifold edges
            if remove_non_manifold_edges:
                mesh = remove_nme(mesh)

            # Yield mesh
            if return_filename:
                yield mesh, shape_path
            else:
                yield mesh


def up_shapenet_generator_unpacked(shapenet_path,
                                   return_filename=False,
                                   remove_non_manifold_edges=False,
                                   down_sample=None,
                                   synset_ids=None):
    """Yields shapenet shapes from unpacked zip-files."""
    # Synset (WordNet): Set of synonyms
    if synset_ids is None:
        synset_ids = os.listdir(shapenet_path)

    for synset_id in synset_ids:
        content = os.listdir(f"{shapenet_path}/{synset_id[:-4]}")
        content.sort()
        for shape_id in content:
            shape_id = f"{shapenet_path}/{synset_id[:-4]}/{shape_id}/models/model_normalized.obj"

            # Load shape
            mesh = trimesh.load_mesh(shape_id)

            # Concat meshes if a scene was loaded
            if type(mesh) == trimesh.scene.Scene:
                mesh = trimesh.util.concatenate([y for y in mesh.geometry.values()])

            # Down-sample mesh
            if down_sample is not None and mesh.faces.shape[0] > down_sample:
                mesh = down_sample_mesh(mesh, down_sample)

            # Remove non-manifold edges
            if remove_non_manifold_edges:
                mesh = remove_nme(mesh)

            # Logging
            non_manifold_edges = np.asarray(mesh.as_open3d.get_non_manifold_edges()).shape[0]
            print(f"{shape_id} | Non-manifold edges: {non_manifold_edges} | vertices: {mesh.vertices.shape[0]}")

            # Yield mesh
            if return_filename:
                yield mesh, shape_id
            else:
                yield mesh
