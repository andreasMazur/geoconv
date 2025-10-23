from geoconv.preprocessing.dgpc.dgpc_util import dgpc_update_step
from geoconv.utils.misc import get_neighbors, get_faces_of_edge

import trimesh
import heapq
import numpy as np
import c_extension


class DgpcSolver:
    def __init__(self, V, F, u_max, eps=1e-5):
        self.triangle_mesh = trimesh.Trimesh(vertices=V, faces=F)
        self.u_max = u_max
        self.eps = eps
        self.edge_cache = {}
        self.proj_cache = {}

    def init_neighborhood(self, source_point):
        """Initializes radial- and angular coordinates of the one-hop neighborhood of a given source point.

        Parameters
        ----------
        source_point: int
            The origin of the chart that shall be calculated.

        Return
        ------
        (np.ndarray, np.ndarray):
            The initialized radial- and angular coordinates.
        """
        # Initialize empty arrays for radial distances and angular directions
        radial_coordinates = np.full((self.triangle_mesh.vertices.shape[0],), fill_value=np.inf)
        angular_coordinates = np.full((self.triangle_mesh.vertices.shape[0],), fill_value=-1.)

        # Get one-hop neighborhood
        neighbors = get_neighbors(source_point, self.triangle_mesh)
        self.one_hop_neighborhood = neighbors

        # Initialize radial distances as Euclidean distances
        neighbor_distances = np.linalg.norm(
            self.triangle_mesh.vertices[neighbors] - self.triangle_mesh.vertices[source_point], axis=-1
        )
        radial_coordinates[neighbors] = neighbor_distances
        radial_coordinates[source_point] = 0.0

        # Initialize initial angular directions randomly (gauge-ambiguity)
        reference_neighbor = neighbors[0]
        rotation_axis = self.triangle_mesh.vertex_normals[source_point]
        vector_a = self.triangle_mesh.vertices[reference_neighbor] - self.triangle_mesh.vertices[source_point]
        for neighbor in neighbors[1:]:  # skip the reference neighbor
            vector_b = self.triangle_mesh.vertices[neighbor] - self.triangle_mesh.vertices[source_point]
            angular_coordinates[neighbor] = c_extension.compute_angle_360(vector_a, vector_b, rotation_axis)
        angular_coordinates[reference_neighbor] = 0.0
        angular_coordinates[source_point] = 0.0

        # Return initial radial- and angular coordinates
        return radial_coordinates, angular_coordinates

    def compute_dgpc(self, source_point):
        """The algorithm for computing Geodesic Polar Coordinate Systems on a triangle mesh.

        The algorithm was published in:
        > [Geodesic polar coordinates on polygonal meshes]
          (https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
        > Melvær, Eivind Lyche, and Martin Reimers.

        Parameters
        ----------
        source_point: int
            The origin of the chart that shall be calculated.

        Return
        ------
        (np.ndarray, np.ndarray):
            The computed radial- and angular coordinates.
        """
        ##########################################################
        # 0.) Initialize one-hop neighborhood around source point
        ##########################################################
        radial_coordinates, angular_coordinates = self.init_neighborhood(source_point)

        ################################################
        # 1.) Initialize min-heap over radial distances
        ################################################
        candidates = []
        one_hop_neighbors = get_neighbors(source_point, self.triangle_mesh)
        for neighbor in [x for x in one_hop_neighbors if x != source_point]:
            candidates.append((radial_coordinates[neighbor], neighbor))
        heapq.heapify(candidates)

        ##########################
        # 2.) Enter the iteration
        ##########################
        while candidates:
            # Get vertex from min-heap that is closest to GPC-system origin
            j_dist, j = heapq.heappop(candidates)
            j_neighbors = get_neighbors(j, self.triangle_mesh)

            # Exclude one-hop neighborhood or are the vertex j itself from updatable vertices
            j_neighbors = [x for x in j_neighbors if x not in self.one_hop_neighborhood + [j]]

            for i in j_neighbors:
                # For the update of vertex `i`, we need to select a third vertex `k` that can be used together
                # with `i` and `j` to calculate a coordinate update for vertex `i`.
                sorted_edge = np.sort([i, j])
                if tuple(sorted_edge) in self.edge_cache.keys():
                    i_j_faces = self.edge_cache[tuple(sorted_edge)]
                else:
                    i_j_faces = get_faces_of_edge(sorted_edge, self.triangle_mesh)
                    self.edge_cache[tuple(sorted_edge)] = i_j_faces

                # Compute the (updated) geodesic distance `new_u_i` and angular coordinate of the i-th neighbor from the
                # closest vertex in the min-heap to the source point of the GPC-system
                updates_list = []
                for face in i_j_faces:
                    # Determine third vertex k of the face
                    k = [k for k in face if k not in [i, j]][0]

                    # Skip if radial or angular coordinates of vertex j or k are not defined
                    if (
                            radial_coordinates[j] == np.inf or
                            angular_coordinates[j] == -1. or
                            radial_coordinates[k] == np.inf or
                            angular_coordinates[k] == -1.
                    ):
                        continue

                    # Compute update step
                    u_i, theta_i = dgpc_update_step(
                        vertex_i_3d=self.triangle_mesh.vertices[i],
                        vertex_j_3d=self.triangle_mesh.vertices[j],
                        vertex_k_3d=self.triangle_mesh.vertices[k],
                        u_j=radial_coordinates[j],
                        u_k=radial_coordinates[k],
                        theta_j=angular_coordinates[j],
                        theta_k=angular_coordinates[k],
                        use_c=True
                    )
                    updates_list.append((u_i, theta_i))
                if len(updates_list) > 0:
                    # Select the smallest update among all considered faces
                    new_u_i, new_theta_i = min(updates_list, key=lambda x: x[0])

                    # Euclidean distance is the lower bound for the radial coordinate
                    euc_dist_s = np.linalg.norm(
                        self.triangle_mesh.vertices[i] - self.triangle_mesh.vertices[source_point]
                    )
                    if euc_dist_s > new_u_i + self.eps:
                        continue

                    # As long as new_u_i is smaller than the current radial coordinate of vertex i
                    # and u_max, we update it
                    if new_u_i <= self.u_max and new_u_i < radial_coordinates[i]:
                        radial_coordinates[i] = new_u_i
                        angular_coordinates[i] = new_theta_i

                        # If the new radial coordinate is still smaller than u_max, we add it to the min-heap
                        if new_u_i < self.u_max:
                            # Before adding i to the min-heap, make sure that it's not already in there
                            candidates = [(u_i, idx) for u_i, idx in candidates if idx != i]
                            heapq.heappush(candidates, (new_u_i, i))
        return radial_coordinates, angular_coordinates
