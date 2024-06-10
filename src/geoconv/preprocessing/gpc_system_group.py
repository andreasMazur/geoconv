from geoconv.preprocessing.gpc_system import GPCSystem
from geoconv.preprocessing.gpc_system_utils import compute_distance_and_angle
from geoconv.utils.misc import get_neighbors

from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import warnings
import heapq
import os
import json


class GPCSystemGroup:
    def __init__(self, object_mesh, eps=0.000001, use_c=True, processes=1):
        self.object_mesh = object_mesh
        self.eps = eps
        self.use_c = use_c
        self.processes = processes
        self.object_mesh_gpc_systems = None

    def compute(self, u_max=.04):
        """Computes geodesic polar coordinates for all vertices within an object mesh.

        Parameters
        ----------
        u_max: float
            The maximal radius for each GPC-system.
        """
        n_vertices = self.object_mesh.vertices.shape[0]
        vertex_indices = np.arange(n_vertices)
        with Pool(self.processes) as p:
            gpc_systems = p.starmap(
                self.compute_gpc_system,
                tqdm(
                    [(vi, u_max) for vi in vertex_indices],
                    total=n_vertices,
                    postfix="Computing GPC-systems"
                )
            )
        self.object_mesh_gpc_systems = np.array(gpc_systems).flatten()

    def compute_gpc_system(self, source_point, u_max, gpc_system=None, plot_path="", max_iter=10_000):
        """Computes local GPC for one given source point.

        This method implements the algorithm of:
        > [Geodesic polar coordinates on polygonal meshes]
          (https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
        > Melvær, Eivind Lyche, and Martin Reimers.

        Parameters
        ----------
        source_point: int
            The index of the source point around which a window (GPC-system) shall be established
        u_max: float
            The maximal distance (e.g. radius of the patch) which a vertex may have to `source_point`
        gpc_system: GPCSystem
            A GPC-system that has been computed previously on the same mesh. Its caches can be re-used
            which saves computation time.
        plot_path: bool
            If given, the update steps will be plotted and stored at the given path.
        max_iter: int
            The maximum amount of update steps.

        Returns
        -------
        GPCSystem:
            A GPC-system.
        """
        ########################
        # Initialize GPC-system
        ########################
        if gpc_system is None:
            gpc_system = GPCSystem(source_point, self.object_mesh, use_c=True)
        else:
            gpc_system.soft_clear(source_point)
        # Check whether initialization distances are larger than given max-radius
        check_array = np.array([x for x in gpc_system.radial_coordinates if not np.isinf(x)])
        if check_array.max() > u_max:
            warnings.warn(
                f"You chose a 'u_max' to be smaller then {check_array.max()}, which has been seen as an initialization"
                f" length for a GPC-system. Current GPC-system will only contain initialization vertices.",
                RuntimeWarning
            )

        ############################################
        # Initialize min-heap over radial distances
        ############################################
        candidates = []
        for neighbor in get_neighbors(source_point, self.object_mesh):
            candidates.append((gpc_system.radial_coordinates[neighbor], neighbor))
        heapq.heapify(candidates)

        ###################################
        # Algorithm to compute GPC-systems
        ###################################
        plot_number = 0
        iteration = 0
        while candidates:
            # Get vertex from min-heap that is closest to GPC-system origin
            j_dist, j = heapq.heappop(candidates)
            j_neighbors = get_neighbors(j, self.object_mesh)
            j_neighbors = [j for j in j_neighbors if j != source_point]
            for i in j_neighbors:
                # Compute the (updated) geodesic distance `new_u_i` and angular coordinate of the i-th neighbor from the
                # closest vertex in the min-heap to the source point of the GPC-system
                new_u_i, new_theta_i, k_vertices = compute_distance_and_angle(
                    i,
                    j,
                    gpc_system,
                    self.use_c,
                    rotation_axis=self.object_mesh.vertex_normals[source_point]
                )
                # Break if max-iteration has been exceeded
                iteration += 1
                if iteration >= max_iter:
                    break
                # In difference to the original pseudocode, we add 'new_u_i < u_max' to this IF-query
                # to ensure that the radial coordinates do not exceed 'u_max'.
                if new_u_i < u_max and gpc_system.radial_coordinates[i] / new_u_i > 1 + self.eps:
                    if plot_path:
                        if gpc_system.update(
                            i, new_u_i, new_theta_i, j, k_vertices, plot_name=f"{plot_path}/{plot_number}"
                        ):
                            heapq.heappush(candidates, (new_u_i, i))
                            plot_number += 1
                    else:
                        if gpc_system.update(i, new_u_i, new_theta_i, j, k_vertices):
                            heapq.heappush(candidates, (new_u_i, i))
            # Break if max-iteration has been exceeded
            if iteration >= max_iter:
                print(f"**** GPC-algorithm: Source Point {source_point} - Maximum iterations reached! ****")
                break
        return gpc_system

    def save(self, path, compressed=True):
        """Saves all GPC-systems.

        Parameters
        ----------
        path: str
            The path to where the GPC-system group shall be saved.
        compressed: bool
            Whether to store the coordinates and properties of a GPC-system in 5 files instead of individual
            GPC-system subdirectories.
        """
        # Create saving directory
        os.makedirs(path, exist_ok=True)

        ####################################
        # Save compressed GPC-system format
        ####################################
        if compressed:
            # Initialize dictionaries which shall be saved
            radial_dict, angular_dict = {}, {}
            x_coordinates_dict, y_coordinates_dict = {}, {}
            gpc_properties_dict = {}

            # Collect information about all GPC-systems
            for gpc_system_idx, gpc_system in tqdm(
                    enumerate(self.object_mesh_gpc_systems), postfix="Saving GPC-systems.."
            ):
                # Store current GPC-system properties in saving dictionaries
                radial_dict[f"{gpc_system_idx}"] = gpc_system.radial_coordinates
                angular_dict[f"{gpc_system_idx}"] = gpc_system.angular_coordinates
                x_coordinates_dict[f"{gpc_system_idx}"] = gpc_system.x_coordinates
                y_coordinates_dict[f"{gpc_system_idx}"] = gpc_system.y_coordinates
                gpc_properties_dict[f"{gpc_system_idx}"] = {
                    "edges": {int(k): v for k, v in gpc_system.edges.items()},
                    "faces": {f"{k1},{k2}": v for (k1, k2), v in gpc_system.faces.items()},
                    "source_point": int(gpc_system.source_point)
                }

            # Save all GPC-system properties in one corresponding file
            np.savez(f"{path}/radial_coordinates.npz", **radial_dict)
            np.savez(f"{path}/angular_coordinates.npz", **angular_dict)
            np.savez(f"{path}/x_coordinates.npz", **x_coordinates_dict)
            np.savez(f"{path}/y_coordinates.npz", **y_coordinates_dict)
            with open(f"{path}/properties.json", "w") as properties_file:
                json.dump(gpc_properties_dict, properties_file)
        ###############################################################################
        # Load regular GPC-system format (each GPC-system has individual subdirectory)
        ###############################################################################
        else:
            for gpc_system_idx, gpc_system in tqdm(
                    enumerate(self.object_mesh_gpc_systems), postfix="Saving GPC-systems.."
            ):
                gpc_system.save(f"{path}/{gpc_system_idx}")

    def load(self, path, load_compressed=True):
        """Loads GPC-systems.

        Parameters
        ----------
        path: str
            The path from where to load the GPC-systems.
        load_compressed: bool
            Whether to load from a compressed store-format (cf. self.save()).
        """
        # Initialize GPC-system list
        gpc_systems = []

        ####################################
        # Load compressed GPC-system format
        ####################################
        if load_compressed:
            # Load all GPC-systems from compressed file format
            radial_coordinates_dict = np.load(f"{path}/radial_coordinates.npz")
            angular_coordinates_dict = np.load(f"{path}/angular_coordinates.npz")
            x_coordinates_dict = np.load(f"{path}/x_coordinates.npz")
            y_coordinates_dict = np.load(f"{path}/y_coordinates.npz")
            with open(f"{path}/properties.json", "r") as properties_file:
                properties = json.load(properties_file)

            # Initialize all GPC-systems using the information stored within the compressed files
            for source_point in tqdm(range(len(properties.keys())), postfix=f"Loading GPC-systems from: '{path}'"):
                gpc_system = GPCSystem(source_point, self.object_mesh, use_c=True)
                gpc_system.load(from_dict={
                    "angular_coordinates": radial_coordinates_dict[f"{source_point}"],
                    "radial_coordinates": angular_coordinates_dict[f"{source_point}"],
                    "x_coordinates": x_coordinates_dict[f"{source_point}"],
                    "y_coordinates": y_coordinates_dict[f"{source_point}"],
                    "properties": properties[f"{source_point}"]
                })
                gpc_systems.append(gpc_system)
        ###############################################################################
        # Load regular GPC-system format (each GPC-system has individual subdirectory)
        ###############################################################################
        else:
            # Read root-directory of all GPC-systems
            gpc_system_directories = os.listdir(path)
            gpc_system_directories.sort(key=lambda fn: int(fn))

            # Iterate over root-directory content and load each GPC-system individually from file-system
            for source_point, gpc_system_directory in tqdm(
                    enumerate(gpc_system_directories), postfix=f"Loading GPC-systems from: '{path}'"
            ):
                gpc_system = GPCSystem(source_point, self.object_mesh, use_c=True)
                gpc_system.load(f"{path}/{gpc_system_directory}")
                gpc_systems.append(gpc_system)

        # Remember GPC-systems
        self.object_mesh_gpc_systems = np.array(gpc_systems)
