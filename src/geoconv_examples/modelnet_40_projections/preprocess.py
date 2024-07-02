from geoconv.utils.data_generator import zip_file_generator

import trimesh


def preprocess(modelnet_path, output_path, down_sample):
    shape_generator = zip_file_generator(
        modelnet_path,
        file_type="off",
        return_filename=True,
        manifold_plus_executable="/home/andreas/programs/ManifoldPlus/build/manifold",
        target_amount_faces=3000,
        remove_non_manifold_edges=True,
        normalize=True
    )
    for shape, shape_path in shape_generator:
        vertices = trimesh.sample.sample_surface_even(shape, count=2000)[0]
        pc = trimesh.PointCloud(vertices=vertices)
        pc.show()


if __name__ == "__main__":
    preprocess(
        modelnet_path="/home/andreas/Uni/datasets/ModelNet40.zip",
        output_path="/home/andreas/Uni/datasets/ModelNet40_rescaled_meshes",
        down_sample=1500
    )
