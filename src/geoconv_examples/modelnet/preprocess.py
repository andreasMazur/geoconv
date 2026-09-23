from geoconv.preprocessing.atlas import Atlas, load_atlas

import point_cloud_utils as pcu
import zipfile
import os
import trimesh
import io
import numpy as np
import time


FOLDER_TO_NUMBER = {
    "bathtub": 0,
    "bed": 1,
    "chair": 2,
    "desk": 3,
    "dresser": 4,
    "monitor": 5,
    "night_stand": 6,
    "sofa": 7,
    "table": 8,
    "toilet": 9
}


def save_atlas(atlas, mesh_save_path):
    """Convenience function to run Saving on a cluster filesystem that is not always reachable, to save an atlas.

    Parameters
    ----------
    atlas: Atlas
        The atlas to save.
    mesh_save_path: str
        The path to where to save the atlas.

    Returns
    -------
    bool:
        Whether the atlas was saved successfully.
    """
    is_saved = False
    tries = 0
    while not is_saved:
        try:
            atlas.save(mesh_save_path)
            is_saved = True
        except BlockingIOError:
            print(f"Trying to save atlas.. {tries}")
            tries += 1
            time.sleep(1)


def get_atlas(max_chart_radius,
              method,
              normalization_method,
              processes,
              zip_file,
              mesh_filepath,
              mesh_save_path):
    """Computes an atlas for a given shape.

    Parameters
    ----------
    max_chart_radius: float
        The maximum allowed chart radius for any chart.
    method: str
        The charting algorithm to use.
    normalization_method: str
        The distance computation method to use during mesh normalization.
    processes: int
        The amount of concurrent processes for computing charts.
    zip_file: zipfile.ZipFile
        The zip-file object for the ModelNet10 dataset.
    mesh_filepath: str
        The path to the shape within the zip-file of the ModelNet10 dataset, whose charts shall be computed.
    mesh_save_path: str
        The path to where to store the atlas.

    Returns
    -------
    Atlas:
        The Atlas-object for the watertight ModelNet10 shape stored at 'mesh_filepath'.
    """
    old_resolution, resolution = 1_000, 1_000
    did_preprocess = False
    while not did_preprocess:
        # Load mesh with given resolution
        mesh = load_modelnet_mesh(zip_file, mesh_filepath, resolution=resolution)
        try:
            if not os.path.isfile(mesh_save_path):
                # Compute the atlas
                atlas = Atlas(
                    triangle_mesh=mesh,
                    max_radius=max_chart_radius,
                    method=method,
                    normalization_method=normalization_method,
                    processes=processes
                )
                atlas.store_array({"ground_truth": np.array([FOLDER_TO_NUMBER[mesh_filepath.split("/")[1]]])})
                os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
                save_atlas(atlas, mesh_save_path)

                # Indicate that preprocessing was successful
                did_preprocess = True
            else:
                print(f"{mesh_save_path} already exists. Skipping.")
                atlas = load_atlas(mesh_save_path)
                did_preprocess = True
        except RuntimeError:
            # Reduce the resolution in case the preprocessing was not successful
            old_resolution = resolution
            resolution = int(resolution * 9/10)
            print(
                f"Failed to compute GPC-systems for: {mesh_filepath}. "
                f"Reducing mesh resolution: {old_resolution} -> {resolution}."
            )
    return atlas


def load_modelnet_mesh(zip_file, mesh_filepath, resolution=1_000):
    """Loads one shape from the raw ModelNet10 zip file and makes it a manifold mesh.

    Uses the algorithm of:
    > [Robust watertight manifold surface generation method for shapenet models](https://arxiv.org/abs/1802.01698)
    > Jingwei Huang, Hao Su and Leonidas Guibas
    to make meshes watertight for distance calculations.

    Implementation available at:
    > https://fwilliams.info/point-cloud-utils/

    Parameters
    ----------
    zip_file: zipfile.ZipFile
        The zip-file object for the ModelNet10 dataset.
    mesh_filepath: str
        The path within the zip-file of the shape that shall be loaded
    resolution: int
        The number of target number of vertices in the processed manifold mesh. This number is not guaranteed.

    Returns
    -------
    trimesh.Trimesh:
        A manifold mesh.
    """
    # Load the mesh
    mesh = trimesh.load_mesh(io.BytesIO(zip_file.read(mesh_filepath)), file_type="off")

    # Concat meshes if a scene was loaded
    if type(mesh) == trimesh.scene.Scene:
        mesh = trimesh.util.concatenate([y for y in mesh.geometry.values()])

    # Reduce resolution of mesh
    new_vertices, new_faces = pcu.make_mesh_watertight(v=mesh.vertices, f=mesh.faces, resolution=resolution)
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    # Filter non-manifold vertices and faces of watertight mesh
    nm_vertices = np.asarray(mesh.as_open3d.get_non_manifold_vertices())
    faces = np.array(
        [x not in nm_vertices and y not in nm_vertices and z not in nm_vertices for [x, y, z] in mesh.faces]
    )
    mesh = mesh.submesh([faces])[0]

    # Make mesh watertight
    new_vertices, new_faces = pcu.make_mesh_watertight(v=mesh.vertices, f=mesh.faces, resolution=resolution)
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    splitted_mesh = mesh.split()
    if len(splitted_mesh) > 1:
        n_vertices = np.array([m.vertices.shape[0] for m in splitted_mesh])
        return splitted_mesh[n_vertices.argmax()]
    else:
        return mesh


def preprocess_modelnet(zip_path,
                        output_path,
                        template_resolutions,
                        max_chart_radius,
                        method="hdm",
                        normalization_method="hdm",
                        processes=1,
                        template_radius_aggregation_method="mean"):
    """Preprocess ModelNet40 shapes.

    Parameters
    ----------
    zip_path: str
        The path to the raw ModelNet10 zip-file.
    output_path: str
        The path that points to where the preprocessed dataset will be stored.
    template_resolutions: list
        A list of tuples, each describing the template resolution of a discretized template.
    max_chart_radius: float
        The maximum allowed chart radius for any chart.
    method: str
        The charting algorithm to use.
    normalization_method: str
        The method to use for computing the geodesic diameter during mesh normalization.
    processes: int
        The number of concurrent processes for computing charts and barycentric coordinates.
    template_radius_aggregation_method: str
        Either "mean" or "median". That's the method used to aggregate over all chart radii to determine a template
        radius. This template radius is then used to compute barycentric coordinates.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_content = [f for f in zip_file.namelist() if f.endswith(".off")]
        zip_content.sort()

        # 2.) Prepare output directory
        os.makedirs(output_path, exist_ok=True)

        # 3.) Compute local charts
        gpc_system_radii = []
        for mesh_filepath in zip_content:
            mesh_save_path = f"{output_path}/{mesh_filepath.split('.')[0]}.hdf5"
            print(f"[GPC-systems] Computing GPC-systems for: '{mesh_filepath}'")
            atlas = get_atlas(
                max_chart_radius,
                method,
                normalization_method,
                processes,
                zip_file,
                mesh_filepath,
                mesh_save_path
            )

            # Remember chart radii for BC computation
            gpc_system_radii.extend(atlas.chart_radii.tolist())

        # Filter distorted GPC-system radii (zero radii)
        gpc_system_radii = [r for r in gpc_system_radii if r > 0]

        # 4.) Compute barycentric coordinates
        for mesh_filepath in zip_content:
            # Load atlas
            mesh_save_path = f"{output_path}/{mesh_filepath.split('.')[0]}.hdf5"
            atlas = load_atlas(mesh_save_path)

            # Compute BC for all template resolutions
            for template_resolution in template_resolutions:
                n_radial, n_angular = template_resolution

                # Determine what template radius to use depending on all chart extensions from whole dataset
                if template_radius_aggregation_method == "mean":
                    template_radius = np.mean(gpc_system_radii)
                elif template_radius_aggregation_method == "median":
                    template_radius = np.median(gpc_system_radii)
                else:
                    raise ValueError(
                        f"Unknown aggregation method: {template_radius_aggregation_method}. "
                        f"Select from: ['mean', 'median']."
                    )

                # Compute BC for all template radii
                print(
                    f"[BC computation] Calculating BC "
                    f"'{n_radial, n_angular, template_radius}' for '{mesh_filepath}']"
                )
                atlas.determine_barycentric_coordinates(
                    n_radial=n_radial,
                    n_angular=n_angular,
                    template_radius=template_radius
                )

            # Save atlas
            atlas.save_training_data(mesh_save_path[:-5])

            # Cleanup old atlas file
            os.remove(mesh_save_path)
