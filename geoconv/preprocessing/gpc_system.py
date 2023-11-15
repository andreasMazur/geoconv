from geoconv.preprocessing.barycentric_coordinates import polar_to_cart
from geoconv.utils.misc import get_neighbors, get_faces_of_edge, compute_vector_angle

import c_extension
import numpy as np


class GPCSystem:
    def __init__(self, source_point, object_mesh, faces=None, use_c=True):
        """Compute the initial radial and angular coordinates around a source point and setup caches.

        Angle coordinates are always given w.r.t. some reference direction. The choice of a reference
        direction can be arbitrary. Here, we choose the vector `x - source_point` with `x` being the
        first neighbor return by `get_neighbors` as the reference direction.

        Parameters
        ----------
        source_point: int
            The index of the source point around which a window (GPC-system) shall be established
        object_mesh: trimesh.Trimesh
            A loaded object mesh
        faces:
        use_c: bool
            A flag whether to use the c-extension
        """
        ####################################################################################
        # Initialize face- and edge-cache with one-hop-neighborhood edges from source-point
        ####################################################################################
        self.edges = {-1: []}
        self.faces = faces if faces is not None else {}
        source_point_neighbors = get_neighbors(source_point, object_mesh)
        for neighbor in source_point_neighbors:
            edge, considered_faces = get_faces_of_edge(np.array([source_point, neighbor]), object_mesh)
            edge = list(edge)
            # Add edges to edge-cache
            new_vertex = edge[0] if edge[0] != source_point else edge[1]
            self.edges[source_point].append(edge)
            self.edges[new_vertex] = [edge]
            self.edges[-1].append(edge)
            # Add faces to face-cache
            self.faces[(edge[0], edge[1])] = []
            for face in considered_faces:
                self.faces[(edge[0], edge[1])].append(np.array(face))

        #######################################
        # Calculate initial radial coordinates
        #######################################
        self.radial_coordinates = np.full((object_mesh.vertices.shape[0],), np.inf)
        r3_source_point = object_mesh.vertices[source_point]
        r3_neighbors = object_mesh.vertices[source_point_neighbors]
        self.radial_coordinates[source_point_neighbors] = np.linalg.norm(
            r3_neighbors - np.stack([r3_source_point for _ in range(len(source_point_neighbors))]), ord=2, axis=-1
        )
        self.radial_coordinates[source_point] = .0

        ########################################
        # Calculate initial angular coordinates
        ########################################
        self.angular_coordinates = np.full((object_mesh.vertices.shape[0],), -1.0)
        ref_neighbor = source_point_neighbors[0]
        rotation_axis = object_mesh.vertex_normals[source_point]
        theta_neighbors = np.full((len(source_point_neighbors, )), .0)
        for idx, neighbor in enumerate(source_point_neighbors):
            vector_a = object_mesh.vertices[ref_neighbor] - object_mesh.vertices[source_point]
            vector_b = object_mesh.vertices[neighbor] - object_mesh.vertices[source_point]
            if use_c:
                theta_neighbors[idx] = c_extension.compute_angle_360(vector_a, vector_b, rotation_axis)
            else:
                theta_neighbors[idx] = compute_vector_angle(vector_a, vector_b, rotation_axis)
        self.angular_coordinates[source_point_neighbors] = theta_neighbors
        self.angular_coordinates[source_point] = 0.0

    def add_edge(self, edge):
        """Add an edge to the GPC-system

        Parameters
        ----------
        edge: list
            The edge to add
        """
        edge = list(edge)
        # Check whether node already exists
        if edge[0] not in self.edges.keys():
            self.edges[edge[0]] = []
        # Check if edge was seen once
        if edge not in self.edges[-1]:
            self.edges[-1].append(edge)
        # Check whether edge is already saved under node
        if edge not in self.edges[edge[0]]:
            self.edges[edge[0]].append(edge)

    def update(self, vertex_i, rho_i, theta_i, vertex_j, vertex_k):
        """Add a face to the GPC-system

        Parameters
        ----------
        vertex_i: int
            The updated vertex
        rho_i: float
            The new radial coordinate for `vertex_i`
        theta_i: float
            The new angular coordinate for `vertex_i`
        vertex_j: int
            The first vertex that we used to update the coordinates of `vertex_i`
        vertex_k: int
            The second vertex that we used to update the coordinates of `vertex_i`

        Returns
        -------
        bool:
            Whether the update succeeded, i.e. the update on `vertex_i` did not cause intersections
        """
        # Sort vertex indices such that edge-cache does not store edges twice
        sorted_face = np.sort([vertex_i, vertex_j, vertex_k])

        #############################################
        # Check edges of `vertex_i` on intersections
        #############################################
        updated_face_edges = [
            [sorted_face[0], sorted_face[1]], [sorted_face[1], sorted_face[2]], [sorted_face[0], sorted_face[2]]
        ]
        edges_of_interest = self.edges[vertex_i].copy()
        for edge in updated_face_edges:
            if edge not in edges_of_interest:
                edges_of_interest.append(edge)

        ##########################
        # Check for intersections
        ##########################
        for edge in edges_of_interest:
            if edge[0] == vertex_i:
                edge_fst_vertex = [rho_i, theta_i]
            else:
                edge_fst_vertex = [self.radial_coordinates[edge[0]], self.angular_coordinates[edge[0]]]
            if edge[1] == vertex_i:
                edge_snd_vertex = [rho_i, theta_i]
            else:
                edge_snd_vertex = [self.radial_coordinates[edge[1]], self.angular_coordinates[edge[1]]]
            if self.line_segment_intersection(np.array([edge_fst_vertex, edge_snd_vertex]), np.array(edge)):
                return False

        ################
        # Add new edges
        ################
        for edge in edges_of_interest:
            self.add_edge(edge)

        ###########################
        # Update GPC of `vertex_i`
        ###########################
        self.radial_coordinates[vertex_i] = rho_i
        self.angular_coordinates[vertex_i] = theta_i
        return True

    def line_segment_intersection(self, new_line_segment, new_line_segment_indices):
        """Checks, whether a line segment intersects previously existing ones.

        Implements:
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment

        Parameters
        ----------
        new_line_segment: np.ndarray
            The new line segment in polar coordinates to be checked.
        new_line_segment_indices: np.ndarray
            The indices of the new line segment to be checked.

        Returns
        -------
        bool:
            Whether the new line segment intersect already existing ones.
        """
        # Calculate Cartesian coordinates of new line segment
        x1, y1 = polar_to_cart(angle=new_line_segment[0, 1], scale=new_line_segment[0, 0])
        x2, y2 = polar_to_cart(angle=new_line_segment[1, 1], scale=new_line_segment[1, 0])
        for line_segment in self.edges[-1]:
            if not np.array_equal(new_line_segment_indices, line_segment):
                # Calculate Cartesian coordinates of existing line segment
                theta_2, rho_2 = self.angular_coordinates[line_segment[0]], self.radial_coordinates[line_segment[0]]
                theta_3, rho_3 = self.angular_coordinates[line_segment[1]], self.radial_coordinates[line_segment[1]]
                x3, y3 = polar_to_cart(angle=theta_2, scale=rho_2)
                x4, y4 = polar_to_cart(angle=theta_3, scale=rho_3)
                # Check on line-segment intersection
                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                nominator_1 = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
                nominator_2 = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
                eps = 1e-5
                x = nominator_1 / denominator
                y = nominator_2 / denominator
                if 0. + eps < x < 1. - eps and 0. + eps < y < 1. - eps:
                    return True
        return False

    def visualize(self):
        pass
