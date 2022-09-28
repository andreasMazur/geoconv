import numpy as np
import pyshot
import zipfile
import os
import tqdm
import trimesh
import shutil


def get_included_faces(object_mesh, gpc_system):
    """Retrieves face indices from GPC-system

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        The object mesh
    gpc_system: np.ndarray
        The considered GPC-system

    Returns
    -------
    list:
        The list of face IDs which are included in the GPC-system
    """
    included_face_ids = []

    # Determine vertex IDs that are included in the GPC-system
    gpc_vertex_ids = np.arange(gpc_system.shape[0])[gpc_system[:, 0] != np.inf]

    # Determine what faces are entirely contained within the GPC-system
    for face_id, face in enumerate(object_mesh.faces):
        counter = 0
        for vertex_id in face:
            counter = counter + 1 if vertex_id in gpc_vertex_ids else counter
        if counter == 3:
            included_face_ids.append(face_id)

    return included_face_ids


def exchange_shot(directory, zip_path, percent=0.12, n_bins=16):
    """Exchanges the SHOT vectors in a given dataset with new ones"""

    target_dir = f"{zip_path[:-4]}_updated"
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(target_dir)

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]

    for file in tqdm.tqdm(file_list):
        object_mesh = trimesh.load_mesh(f"{directory}/{file}")

        descriptors = pyshot.get_descriptors(
            np.array(object_mesh.vertices),
            np.array(object_mesh.faces, dtype=np.int64),
            radius=np.sqrt(percent * object_mesh.area / np.pi),
            local_rf_radius=np.sqrt(percent * object_mesh.area / np.pi),
            min_neighbors=3,
            n_bins=n_bins,
            double_volumes_sectors=True,
            use_interpolation=True,
            use_normalization=True,
        ).astype(np.float32)
        np.save(f"{target_dir}/SHOT_{file[:-4]}.npy", descriptors)

    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
