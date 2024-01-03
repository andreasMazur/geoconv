from geoconv.preprocessing.barycentric_coordinates import polar_to_cart
from geoconv.utils.misc import get_neighbors, get_faces_of_edge, compute_vector_angle, gpc_systems_into_cart

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

import c_extension
import numpy as np
import sys


class GPCSystem:
    def __init__(self, source_point, object_mesh, use_c=True, soft_clear=False):
        """Compute the initial radial and angular coordinates around a source point and setup caches.

        Angle coordinates are always given w.r.t. some reference direction. The choice of a reference
        direction can be arbitrary. Here, we choose the vector `x - source_point` with `x` being the
        first neighbor return by `get_neighbors` as the reference direction.

        This class handles two caches which connect GPC-system- with mesh information:
            - Edge-cache: Remembers all edges to a vertex
            - Face-cache: Remembers all faces to a sorted edge

        Parameters
        ----------
        source_point: int
            The index of the source point around which a window (GPC-system) shall be established.
        object_mesh: trimesh.Trimesh
            A loaded object mesh.
        use_c: bool
            A flag whether to use the c-extension.
        soft_clear: bool
            Whether to re-use old edge- and face-caches.
        """
        # Remember the underlying mesh
        self.object_mesh = object_mesh
        self.source_point = source_point
        self.radial_coordinates = np.full((object_mesh.vertices.shape[0],), np.inf)
        self.angular_coordinates = np.full((object_mesh.vertices.shape[0],), -1.0)
        self.x_coordinates = np.full((object_mesh.vertices.shape[0],), np.inf)
        self.y_coordinates = np.full((object_mesh.vertices.shape[0],), np.inf)

        #######################################
        # Calculate initial radial coordinates
        #######################################
        source_point_neighbors = get_neighbors(source_point, object_mesh)
        r3_source_point = object_mesh.vertices[source_point]
        r3_neighbors = object_mesh.vertices[source_point_neighbors]
        self.radial_coordinates[source_point_neighbors] = np.linalg.norm(
            r3_neighbors - np.stack([r3_source_point for _ in range(len(source_point_neighbors))]), ord=2, axis=-1
        )
        self.radial_coordinates[source_point] = .0

        ########################################
        # Calculate initial angular coordinates
        ########################################
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

        ##########################################
        # Calculate initial Cartesian coordinates
        ##########################################
        self.x_coordinates[source_point] = 0.
        self.y_coordinates[source_point] = 0.
        for neighbor in source_point_neighbors:
            x, y = polar_to_cart(angles=self.angular_coordinates[neighbor], scales=self.radial_coordinates[neighbor])
            self.x_coordinates[neighbor] = x
            self.y_coordinates[neighbor] = y

        ####################################################################################
        # Initialize face- and edge-cache with one-hop-neighborhood edges from source-point
        ####################################################################################
        if not soft_clear:
            self.edges = {-1: []}
            self.faces = {(-1, -1): []}
        self.edges[source_point] = []
        for neighbor in source_point_neighbors:
            edge, considered_faces = get_faces_of_edge(np.array([source_point, neighbor]), object_mesh)
            edge = list(edge)
            # Add edges to edge-cache
            self.add_edge(edge)
            # Add faces to face-cache
            for face in considered_faces:
                self.add_face(face)

    def soft_clear(self, source_point, use_c=True):
        """Reset radial- and angular coordinates, keep underlying mesh and edge- and face-caches.

        Parameters
        ----------
        source_point: int
            The new source point
        use_c: bool
            A flag whether to use the c-extension
        """
        self.__init__(source_point, self.object_mesh, use_c=use_c, soft_clear=True)

    def add_edge(self, edge):
        """Add an edge to the GPC-system

        Parameters
        ----------
        edge: list
            The edge to add
        """
        if np.inf in [self.x_coordinates[edge[0]], self.x_coordinates[edge[1]]]:
            raise RuntimeError(f"Edge {edge} lacks GPC: {[self.x_coordinates[edge[0]], self.x_coordinates[edge[1]]]}")

        edge = list(np.sort(edge))
        # Check if edge was seen once
        if edge not in self.edges[-1]:
            self.edges[-1].append(edge)
        for vertex in edge:
            # Check whether node already exists
            if vertex not in self.edges.keys():
                self.edges[vertex] = []
            # Check whether edge is already saved under node
            if edge not in self.edges[vertex]:
                self.edges[vertex].append(edge)

    def add_face(self, face):
        """Add a face to the GPC-system

        Parameters
        ----------
        face: np.ndarray
            The face to add
        """
        face = list(np.sort(face))
        if face not in self.faces[(-1, -1)]:
            self.faces[(-1, -1)].append(face)
        face_edges = [
            [face[0], face[1]], [face[1], face[2]], [face[0], face[2]]
        ]
        for edge in face_edges:
            if (edge[0], edge[1]) not in self.faces.keys():
                self.faces[(edge[0], edge[1])] = [face]
            elif face not in self.faces[(edge[0], edge[1])]:
                self.faces[(edge[0], edge[1])].append(face)

            # Recursively check all edges on whether their 2nd face is entirely describable with GPCs
            for new_face in get_faces_of_edge(edge, self.object_mesh)[1]:
                new_face = np.sort(new_face)
                # If all face coordinates are known and face has not been seen, then update GPC-system with `new_face`
                if (not np.array_equal(new_face, face)
                    and not np.any(np.isinf(self.radial_coordinates[new_face]))
                    and list(new_face) not in self.faces[(-1, -1)]):
                    self.update(
                        new_face[0],
                        self.radial_coordinates[new_face[0]],
                        self.angular_coordinates[new_face[0]],
                        new_face[1],
                        [new_face[2]],
                        update=False
                    )

    def update(self, vertex_i, rho_i, theta_i, vertex_j, k_vertices, plot_name="", update=True):
        """Update the GPC-system while preventing to edge intersections

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
        k_vertices: list
            The list of second vertices that could have potentially been used to update the coordinates of `vertex_i`
        plot_name: string
            If given, the update step is plotted and stored under the given name.
        update: bool
            Whether to update the stored coordinates of `vertex_i`.

        Returns
        -------
        bool:
            Whether the update succeeded, i.e. the update on `vertex_i` did not cause intersections
        """
        for vertex_k in [k for k in k_vertices if not np.isinf(self.radial_coordinates[k])]:
            # Sort vertex indices such that edge-cache does not store edges twice
            sorted_face = np.sort([vertex_i, vertex_j, vertex_k])

            ###############################################################################################
            # Collect all edges of `vertex_i` that will be added or are captured by the current GPC-system
            ###############################################################################################
            updated_face_edges = [
                [sorted_face[0], sorted_face[1]], [sorted_face[1], sorted_face[2]], [sorted_face[0], sorted_face[2]]
            ]

            if vertex_i in self.edges.keys():
                edges_of_interest = self.edges[vertex_i].copy()
            else:
                self.edges[vertex_i] = []
                edges_of_interest = []

            for edge in updated_face_edges:
                if edge not in edges_of_interest:
                    edges_of_interest.append(edge)

            ##########################################
            # Check collected edges for intersections
            ##########################################
            x, y = polar_to_cart(angles=theta_i, scales=rho_i)
            for edge in edges_of_interest:
                if edge[0] == vertex_i:
                    edge_fst_vertex = [x, y]
                else:
                    edge_fst_vertex = [self.x_coordinates[edge[0]], self.y_coordinates[edge[0]]]

                if edge[1] == vertex_i:
                    edge_snd_vertex = [x, y]
                else:
                    edge_snd_vertex = [self.x_coordinates[edge[1]], self.y_coordinates[edge[1]]]

                if self.line_segment_intersection(edge_fst_vertex, edge_snd_vertex):
                    # Return 'False' to indicate failed update due to intersection
                    return False

            ###########################
            # Update GPC of `vertex_i`
            ###########################
            if update:
                self.radial_coordinates[vertex_i] = rho_i
                self.angular_coordinates[vertex_i] = theta_i

                self.x_coordinates[vertex_i] = x
                self.y_coordinates[vertex_i] = y

            ################
            # Add new edges
            ################
            for edge in edges_of_interest:
                self.add_edge(edge)

            ###############
            # Add new face
            ###############
            self.add_face(sorted_face)

        if plot_name:
            self.plot([vertex_i, vertex_j], plot_name)

        # Return 'True' to indicate successful update
        return True

    def plot(self, new_line, save_name):
        """Plots the currently captured faces and highlights the updated line and vertex.

        new_line: list
            The vertex indices of the updated line.
        save_name: str
            The path under which the plot is saved.
        """
        fig, ax = plt.subplots(1, 1)
        x_new_0 = self.x_coordinates[new_line[0]]
        y_new_0 = self.y_coordinates[new_line[0]]
        x_new_1 = self.x_coordinates[new_line[1]]
        y_new_1 = self.y_coordinates[new_line[1]]
        ax.plot([x_new_0, x_new_1], [y_new_0, y_new_1], color="blue", linewidth=2., label="Updated Line")
        ax.scatter([x_new_0], [y_new_0], color="blue", s=50., label="Updated Vertex")

        all_coordinates = set()
        for face in self.faces[(-1, -1)]:
            x0 = self.x_coordinates[face[0]]
            y0 = self.y_coordinates[face[0]]

            x1 = self.x_coordinates[face[1]]
            y1 = self.y_coordinates[face[1]]

            x2 = self.x_coordinates[face[2]]
            y2 = self.y_coordinates[face[2]]

            coordinates = [(x0, y0), (x1, y1), (x2, y2)]
            for c in coordinates:
                all_coordinates.add(c)

            polygon = Polygon(np.array(coordinates), alpha=.4, edgecolor="red")
            ax.add_patch(polygon)

        all_coordinates = np.array([list(c) for c in all_coordinates])
        ax.set_xlim(all_coordinates[:, 0].min(), all_coordinates[:, 0].max())
        ax.set_ylim(all_coordinates[:, 1].min(), all_coordinates[:, 1].max())

        plt.savefig(f"{save_name}.svg")
        plt.legend(loc="upper right")
        plt.close()

    def line_segment_intersection(self, edge_fst_vertex, edge_snd_vertex):
        """Checks, whether a line segment intersects previously existing ones.

        Implements:
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment

        Parameters
        ----------
        edge_fst_vertex: np.ndarray
            The cartesian coordinates of the first vertex of the line to check
        edge_snd_vertex: np.ndarray
            The cartesian coordinates of the second vertex of the line to check

        Returns
        -------
        bool:
            Whether the new line segment intersect already existing ones.
        """
        x1, y1 = edge_fst_vertex[0], edge_fst_vertex[1]
        x2, y2 = edge_snd_vertex[0], edge_snd_vertex[1]
        all_edges = np.array(self.edges[-1])
        xs3, ys3 = self.x_coordinates[all_edges[:, 0]], self.y_coordinates[all_edges[:, 0]]
        xs4, ys4 = self.x_coordinates[all_edges[:, 1]], self.y_coordinates[all_edges[:, 1]]

        denominators = (x1 - x2) * (ys3 - ys4) - (y1 - y2) * (xs3 - xs4)
        if 0. in denominators:
            denominators += sys.float_info.min
        nominators_1 = (x1 - xs3) * (ys3 - ys4) - (y1 - ys3) * (xs3 - xs4)
        nominators_2 = (x1 - xs3) * (y1 - y2) - (y1 - ys3) * (x1 - x2)

        xs = nominators_1 / denominators
        ys = nominators_2 / denominators

        eps = 1e-5
        return np.any(
            np.logical_and(np.logical_and(0. + eps < xs, xs < 1. - eps), np.logical_and(0. + eps < ys, ys < 1. - eps))
        )

    def get_gpc_system(self):
        """Return the GPC-system as one numpy array.

        An array `self.radial_coordinates` of radial coordinates from the source point to other points in the object
        mesh. An array `self.angular_coordinates` of angular coordinates of neighbors from `source_point` in its window.
        """
        return np.stack([self.radial_coordinates, self.angular_coordinates], axis=1)

    def get_gpc_triangles(self, in_cart=False):
        """Return all triangles captured by the current GPC-system.

        Parameters
        ----------
        in_cart: bool
            Whether to translate geodesic polar coordinates into cartesian.
        """
        gpc_system_triangles = self.get_gpc_system()
        gpc_system_triangles = gpc_system_triangles[np.array(self.faces[(-1, -1)])]
        if in_cart:
            return gpc_systems_into_cart(gpc_system_triangles)
        else:
            return gpc_system_triangles
