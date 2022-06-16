import trimesh
import pyshot
import numpy as np


def main():

    sample_scan = "./MPI-FAUST/training/scans/tr_scan_000.ply"
    sample_registration = "./MPI-FAUST/training/registrations/tr_reg_000.ply"

    #################
    # Data as meshes
    #################
    # Scan mesh
    mesh = trimesh.load_mesh(sample_scan)
    mesh.show()
    points, face_indices = trimesh.sample.sample_surface_even(mesh, 6_000)
    sub_mesh = mesh.submesh([face_indices])[0]
    sub_mesh.show()
    # reg_mesh = trimesh.load_mesh(sample_registration)
    # reg_mesh.show()

    # There is one SHOT-descriptor for every node in the polygon mesh
    # 2D array: (#nodes, #64 + (#bins - 1) * 32), assuming at least one bin
    # Original SHOT-Paper suggested a binning of 32. However, see question below..
    descr = pyshot.get_descriptors(
        np.array(sub_mesh.vertices),
        np.array(sub_mesh.faces, dtype=np.int64),
        radius=100,
        local_rf_radius=.0,
        min_neighbors=0,
        n_bins=32  # How is it, that those descriptors yield 1056 (64 + 31 * 32)-sized vectors?
    )
    # print()

if __name__ == "__main__":
    main()
