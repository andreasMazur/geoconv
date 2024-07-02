from geoconv.preprocessing.wrapper import compute_gpc_systems_wrapper, compute_bc_wrapper
from geoconv.utils.data_generator import zip_file_generator

import shutil


def preprocess(modelnet_path,
               output_path,
               manifold_plus_executable,
               down_sample,
               processes,
               class_names=None,
               zip_when_done=True,
               compute_gpc=True,
               compute_bc=True):
    assert compute_gpc or compute_bc, "You must either set 'compute_gpc' or 'compute_bc' to 'True'."

    ######################
    # Compute GPC-systems
    ######################
    if compute_gpc:
        # Initialize shape generator
        shape_generator = zip_file_generator(
            modelnet_path,
            file_type="off",
            manifold_plus_executable=manifold_plus_executable,
            target_amount_faces=down_sample,
            return_filename=True,
            shape_path_contains=class_names,
            normalize=False  # normalize during GPC-system computation to store original geodesic diameter
        )

        # Compute GPC-systems
        for shape, shape_path in shape_generator:
            print(f"*** Preprocessing: '{shape_path}'")
            # Remove file-ending from folder name
            output_dir = f"{output_path}/{shape_path}"[:-4]
            if class_names is None:
                compute_gpc_systems_wrapper(shape, output_dir, processes=processes, scale=0.11)
            elif shape_path.split("/")[1] in class_names:
                compute_gpc_systems_wrapper(shape, output_dir, processes=processes, scale=0.11)

    ##################################
    # Compute barycentric coordinates
    ##################################
    if compute_bc:
        bc_computed = compute_bc_wrapper(
            preprocess_dir=output_path,
            template_sizes=[(3, 6), (2, 9), (5, 8), (4, 10)],
            scales=[0.75, 1.0, 1.25],
            load_compressed_gpc_systems=True,
            processes=processes,
            shape_path_contains=class_names
        )
        if not bc_computed:
            print("Failed to compute all barycentric coordinates. Skip zipping of intermediate results.")
            return

    #############################
    # Zip preprocessed directory
    #############################
    if zip_when_done:
        print("Zipping..")
        shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
        shutil.rmtree(output_path)
        print("Done.")
