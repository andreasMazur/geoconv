from io import BytesIO

import json
import trimesh
import os
import zipfile


def up_shapenet_generator(shapenet_path, return_filename=False, synset_ids=None, return_properties=False):
    """Yields unprocessed shapenet shapes directly from the zip-files."""
    # Synset (WordNet): Set of synonyms
    if synset_ids is None:
        synset_ids = [f for f in os.listdir(shapenet_path) if f[-3:] == "zip"]
    else:
        # Check for file ending
        synset_ids = [synset_id if synset_id[-3:] == "zip" else f"{synset_id}.zip" for synset_id in synset_ids]

    for synset_id in synset_ids:
        synset_ids_zip = zipfile.ZipFile(f"{shapenet_path}/{synset_id}", "r")
        zip_content = [fn for fn in synset_ids_zip.namelist() if fn[-3:] == "obj"]
        zip_content.sort()
        for shape_path in zip_content:
            # Load shape
            shape = trimesh.load_mesh(BytesIO(synset_ids_zip.read(shape_path)), file_type="obj")

            # Concat meshes if a scene was loaded
            if type(shape) == trimesh.scene.Scene:
                shape = trimesh.util.concatenate([y for y in shape.geometry.values()])

            # Yield mesh
            if return_properties:
                properties_path = f"{'/'.join(shape_path.split('/')[:-1])}/preprocess_properties.json"
                with synset_ids_zip.open(properties_path) as properties_file:
                    properties = json.load(properties_file)
                if return_filename:
                    yield shape, shape_path, properties
                else:
                    yield shape, properties
            else:
                if return_filename:
                    yield shape, shape_path
                else:
                    yield shape


def up_unpacked_shapenet_generator(shapenet_path, return_filename=False, synset_ids=None):
    # Synset (WordNet): Set of synonyms
    if synset_ids is None:
        synset_ids = os.listdir(shapenet_path)

    for synset_id in synset_ids:
        synset_path = f"{shapenet_path}/{synset_id}/{synset_id}"
        content = [f"{synset_path}/{shape_id}/models/model_normalized.obj" for shape_id in os.listdir(synset_path)]
        content.sort()
        for shape_path in content:
            shape = trimesh.load_mesh(shape_path)

            # Concat meshes if a scene was loaded
            if type(shape) == trimesh.scene.Scene:
                shape = trimesh.util.concatenate([y for y in shape.geometry.values()])

            # Yield mesh
            if return_filename:
                # Return relative path of shape in dataset
                yield shape, "/".join(shape_path.split("/")[-5:])
            else:
                yield shape
