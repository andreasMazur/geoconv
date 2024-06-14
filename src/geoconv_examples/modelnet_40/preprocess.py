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
            down_sample=down_sample,
            return_filename=True,
            shape_path_contains=class_names
        )

        # Compute GPC-systems
        for shape, shape_path in shape_generator:
            print(f"*** Preprocessing: '{shape_path}'")
            # Remove file-ending from folder name
            output_dir = f"{output_path}/{shape_path}"[:-4]
            if class_names is None:
                compute_gpc_systems_wrapper(shape, output_dir, processes=processes)
            elif shape_path.split("/")[1] in class_names:
                compute_gpc_systems_wrapper(shape, output_dir, processes=processes)

    ##################################
    # Compute barycentric coordinates
    ##################################
    if compute_bc:
        compute_bc_wrapper(
            preprocess_dir=output_path,
            template_sizes=[(3, 6), (5, 8)],
            scales=[0.75, 1.0, 1.25],
            load_compressed_gpc_systems=True,
            processes=processes,
            shape_path_contains=class_names
        )

    #############################
    # Zip preprocessed directory
    #############################
    if zip_when_done:
        print("Zipping..")
        shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
        shutil.rmtree(output_path)
        print("Done.")
