from io import BytesIO

import trimesh
import os
import zipfile


def up_shapenet_generator(shapenet_path, return_filename=False, synset_ids=None):
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

            # Yield mesh
            if return_filename:
                yield mesh, shape_path
            else:
                yield mesh
