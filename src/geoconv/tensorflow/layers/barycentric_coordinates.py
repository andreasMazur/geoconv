from geoconv.preprocessing.barycentric_coordinates import create_template_matrix

import tensorflow as tf


def group_neighborhoods(coordinates, radius):
    """Finds and groups neighborhoods according to the number of neighbors they contain.

    Parameters
    ----------
    coordinates: tf.Tensor
        All vertices of a mesh.
    radius: float
        The radius of a neighborhood.

    Returns
    -------
    (tf.Tensor, tf.Tensor):
        The indices of the respective origins and the indices of the contained vertices.
    """
    # 1.) For each vertex determine local sets of neighbors
    squared_norm = tf.einsum("ij,ij->i", coordinates, coordinates)
    vertices_in_range = tf.sqrt(
        tf.abs(squared_norm - 2 * tf.einsum("ik,jk->ij", coordinates, coordinates) + squared_norm)
    ) <= radius

    # 2.) Get vertex indices of neighbors
    indices = tf.where(vertices_in_range)
    neighborhood_vertex_indices = tf.RaggedTensor.from_value_rowids(values=indices[:, 1], value_rowids=indices[:, 0])

    # 4.) Group neighborhoods with equal amount of neighbors
    in_range_counts = tf.math.count_nonzero(vertices_in_range, axis=-1)
    in_range_classes, _ = tf.unique(in_range_counts)

    # 5.) Yield current neighborhood group
    for x in in_range_classes:
        neighborhood_group_mask = in_range_counts == x
        group_origins = tf.where(neighborhood_group_mask)
        group_indices = tf.cast(
            tf.ragged.boolean_mask(neighborhood_vertex_indices, neighborhood_group_mask).to_tensor(), tf.int32
        )
        yield group_origins, group_indices


@tf.function
def disambiguate_axes(neighborhood_vertices, eigen_vectors):
    """Disambiguate axes returned by local Eigenvalue analysis.

    Disambiguation follows the formal procedure as described in:
    > [Salti, Samuele, Federico Tombari, and Luigi Di Stefano. "SHOT: Unique signatures of histograms for surface and
     texture description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhood_vertices: tf.Tensor
        The vertices of the neighborhoods.
    eigen_vectors: tf.Tensor
        The Eigenvectors of all neighborhoods for one dimension, i.e. it has size (#neighborhoods, 3).
        E.g. the x-axes.

    Returns
    -------
    tf.Tensor:
        The disambiguated Eigenvalues.
    """
    neg_eigen_vectors = -eigen_vectors
    ev_count = tf.math.count_nonzero(tf.einsum("nvk,nk->nv", neighborhood_vertices, eigen_vectors) >= 0, axis=-1)
    ev_neg_count = tf.math.count_nonzero(tf.einsum("nvk,nk->nv", neighborhood_vertices, -eigen_vectors) > 0., axis=-1)
    return tf.gather(
        tf.stack([neg_eigen_vectors, eigen_vectors], axis=1), tf.cast(ev_count >= ev_neg_count, tf.int32), batch_dims=1
    )


@tf.function
def shot_lrf(neighborhood_vertices, neighborhood_origins, radius):
    """Computes SHOT local reference frames.

    SHOT computation was introduced in:
    > [Salti, Samuele, Federico Tombari, and Luigi Di Stefano. "SHOT: Unique signatures of histograms for surface and
     texture description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhood_vertices: tf.Tensor
        The vertices of the neighborhoods.
    neighborhood_origins: tf.Tensor
        The origins of the neighborhoods.
    radius: float
        The radius of the neighborhoods.

    Returns
    -------
    tf.Tensor:
        Local reference frames for all given neighborhoods.
    """
    ###########################
    # 1.) Compute Eigenvectors
    ###########################
    # Shift neighborhoods into origin
    neighborhood_vertices = neighborhood_vertices - neighborhood_origins

    # Calculate neighbor distances to origin
    radius_dist = tf.abs(radius - tf.sqrt(tf.einsum("ijk,ijk->ij", neighborhood_vertices, neighborhood_vertices)))

    # Compute weighted covariance matrices
    weighted_cov = tf.reshape(1 / tf.reduce_sum(radius_dist, axis=-1), (-1, 1, 1)) * tf.einsum(
        "nvi,nvj,nv->nij", neighborhood_vertices, neighborhood_vertices, radius_dist
    )

    ########################
    # 2.) Disambiguate axes
    ########################
    # First eigen vector corresponds to smallest eigen value (i.e. plane normal)
    _, eigen_vectors = tf.linalg.eigh(weighted_cov)
    x_axes = disambiguate_axes(neighborhood_vertices, eigen_vectors[:, 2, :])
    z_axes = disambiguate_axes(neighborhood_vertices, eigen_vectors[:, 0, :])
    y_axes = tf.linalg.cross(z_axes, x_axes)

    return tf.stack([z_axes, y_axes, x_axes], axis=1)


def project_neighborhood(neighborhood_vertices, neighborhood_origins, radius):
    """Projects neighborhoods onto local tangent planes.

    Parameters
    ----------
    neighborhood_vertices: tf.Tensor
        The vertices of the neighborhoods.
    neighborhood_origins: tf.Tensor
        The origins of the neighborhoods.
    radius: float
        The radius of the neighborhoods.

    Returns
    -------
    tf.Tensor:
        The projected neighborhoods.
    """
    # Compute local reference frames (includes tangent planes)
    lrf = shot_lrf(neighborhood_vertices, neighborhood_origins, radius)

    # Projections into planes of cf_normals (lrf[:, 0, :] = plane normals / z-axes)
    neighborhood_vertices -= (
            neighborhood_vertices @ tf.reshape(lrf[:, 0, :], (-1, 3, 1)) * tf.reshape(lrf[:, 0, :], (-1, 1, 3))
    )

    # Basis change from standard Euclidean to gauges
    basis_change = tf.transpose(lrf, perm=[0, 2, 1])
    if tf.reduce_all(tf.linalg.matrix_rank(basis_change) == 3):
        basis_change = tf.linalg.inv(basis_change)
    else:
        # Calculate pseudo-inverse in case Eigenvector matrix is not invertible
        basis_change = tf.linalg.pinv(basis_change)

    # First coordinate corresponds to plane normal. It is zero after projection into plane.
    return tf.einsum("pij,pkj->pki", basis_change, neighborhood_vertices)[:, :, 1:]


class BarycentricCoordinates(tf.keras.layers.Layer):
    def __init__(self, n_radial, n_angular, radius=0.01, template_radius=0.0075):
        super().__init__()
        self.radius = radius
        self.template_radius = template_radius
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template = tf.constant(
            create_template_matrix(
                n_radial=self.n_radial, n_angular=self.n_angular, radius=self.template_radius, in_cart=True
            ),
            dtype=tf.float32
        )

        bc_indices_0, bc_indices_1 = [], []
        for ri in tf.range(self.n_radial):
            for ai in tf.range(self.n_angular):
                for ti in tf.range(3):
                    bc_indices_0.append([ri, ai, ti, 0])
                    bc_indices_1.append([ri, ai, ti, 1])
        self.bc_indices_0 = tf.stack(bc_indices_0)
        self.bc_indices_1 = tf.stack(bc_indices_1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_radial": self.n_radial,
                "b_angular": self.n_angular,
                "radius": self.radius,
                "template_radius": self.template_radius
            }
        )
        return config

    def call(self, coordinates, *args, **kwargs):
        """Calculate barycentric coordinates.

        Parameters
        ----------
        coordinates: tf.Tensor
            3D-coordinates of a mesh. Has shape: (vertices, 3)

        Returns
        -------
        tf.Tensor:
            The barycentric coordinates tensor for an ISC-layer.
            Has shape: (n_vertices, n_radial, n_angular, 3, 2)
        """
        barycentric_coordinates = tf.zeros(shape=(tf.shape(coordinates)[0], self.n_radial, self.n_angular, 3, 2))
        for (group_origins, group_indices) in group_neighborhoods(coordinates, self.radius):
            # Project neighborhoods into tangent planes
            projections = project_neighborhood(
                neighborhood_vertices=tf.gather(coordinates, group_indices),
                neighborhood_origins=tf.gather(coordinates, group_origins),
                radius=self.radius
            )
            group_bc, bc_indices = self.barycentric_coordinates_approx(projections)

            #################################
            # UPDATE BARYCENTRIC COORDINATES
            #################################
            # Indices
            barycentric_coordinates = tf.tensor_scatter_nd_update(
                barycentric_coordinates,
                indices=tf.concat(
                    [
                        tf.tile(tf.cast(group_origins, tf.int32), [tf.shape(self.bc_indices_0)[0], 1]),
                        tf.tile(self.bc_indices_0, [tf.shape(group_origins)[0], 1])
                    ],
                    axis=1
                ),  # Translate group indices to mesh indices
                updates=tf.cast(tf.reshape(tf.gather(group_indices, bc_indices, batch_dims=1), (-1)), tf.float32)
            )

            # Interpolation values
            barycentric_coordinates = tf.tensor_scatter_nd_update(
                barycentric_coordinates,
                indices=tf.concat(
                    [
                        tf.tile(tf.cast(group_origins, tf.int32), [tf.shape(self.bc_indices_1)[0], 1]),
                        tf.tile(self.bc_indices_1, [tf.shape(group_origins)[0], 1])
                    ],
                    axis=1
                ),
                updates=tf.reshape(group_bc, (-1))
            )
        return barycentric_coordinates

    # @tf.function
    def barycentric_coordinates_approx(self, projections):
        """Computes barycentric coordinates within each projected neighborhood.

        Parameters
        ----------
        projections: tf.Tensor
            The projected neighborhoods.

        Returns
        -------
        (tf.Tensor, tf.Tensor):
            The first tensor contains the barycentric coordinates. The second tensor contains neighborhood indices,
            which in the following should be converted to global mesh indices.
        """
        distances = projections - tf.reshape(self.template, (self.n_radial, self.n_angular, 1, 1, 2))

        if tf.shape(projections)[1] < 3:
            return tf.zeros(
                shape=(tf.shape(projections)[0], self.n_radial, self.n_angular, 3)
            ), tf.zeros(
                shape=(tf.shape(projections)[0], self.n_radial, self.n_angular, 3), dtype=tf.int32
            )

        closest_neighbors_indices = tf.transpose(
            tf.argsort(tf.norm(distances, axis=-1), axis=-1)[:, :, :, :3], perm=[2, 0, 1, 3]
        )
        closest_neighbors = tf.gather(tf.reshape(projections, (-1, 2)), closest_neighbors_indices, axis=0)

        v0 = closest_neighbors[:, :, :, 2] - closest_neighbors[:, :, :, 0]
        v1 = closest_neighbors[:, :, :, 1] - closest_neighbors[:, :, :, 0]
        v2 = tf.expand_dims(self.template, axis=0) - closest_neighbors[:, :, :, 0]

        dot00 = tf.einsum("vrai,vrai->vra", v0, v0)
        dot01 = tf.einsum("vrai,vrai->vra", v0, v1)
        dot02 = tf.einsum("vrai,vrai->vra", v0, v2)
        dot11, dot12 = tf.einsum("vrai,vrai->vra", v1, v1), tf.einsum("vrai,vrai->vra", v1, v2)

        denominator = dot00 * dot11 - dot01 * dot01
        point_2_weight = (dot11 * dot02 - dot01 * dot12) / denominator
        point_1_weight = (dot00 * dot12 - dot01 * dot02) / denominator
        point_0_weight = 1 - point_2_weight - point_1_weight
        bc = tf.stack([point_0_weight, point_1_weight, point_2_weight], axis=-1)

        return bc, closest_neighbors_indices
