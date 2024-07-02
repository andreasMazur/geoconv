from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.utils.compute_shot_lrf import group_neighborhoods, shot_lrf

import tensorflow as tf

from geoconv.utils.misc import find_largest_one_hop_dist
from geoconv_examples.stanford_bunny.preprocess_demo import load_bunny


@tf.function
def logarithmic_map(lrfs, neighborhoods):
    # Get tangent plane normals (z-axes of lrfs)
    normals = lrfs[:, 0, :]

    # Compute tangent plane projections (logarithmic map)
    scaled_normals = neighborhoods @ tf.expand_dims(normals, axis=-1) * tf.expand_dims(normals, axis=1)
    projections = neighborhoods - scaled_normals

    # Basis change of neighborhoods into lrf coordinates
    projections = tf.einsum("vij,vnj->vni", tf.linalg.inv(tf.transpose(lrfs, perm=[0, 2, 1])), projections)[:, :, 1:]

    # Preserve Euclidean metric between original vertices (geodesic distance approximation)
    return tf.expand_dims(tf.linalg.norm(neighborhoods, axis=-1), axis=-1) * tf.math.l2_normalize(projections, axis=-1)


class BarycentricCoordinates(tf.keras.layers.Layer):
    def __init__(self, n_radial, n_angular, radius=0.01, template_scale=0.75):
        super().__init__()
        self.radius = radius
        self.template_radius = template_scale * self.radius
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template = tf.constant(
            create_template_matrix(
                n_radial=self.n_radial, n_angular=self.n_angular, radius=self.template_radius, in_cart=True
            ), dtype=tf.float32
        )

    @tf.function
    def call(self, vertices):
        return tf.map_fn(self.call_helper, vertices)

    @tf.function
    def call_helper(self, vertices):
        # 1.) Get vertex-neighborhoods
        # 'neighborhoods': (vertices, n_neighbors, 3)
        neighborhoods, neighborhoods_indices = group_neighborhoods(vertices, self.radius)

        # 2.) Get local reference frames
        # 'lrfs': (vertices, 3, 3)
        lrfs = shot_lrf(neighborhoods, self.radius)

        # 3.) Project neighborhoods into their lrfs using the logarithmic map
        # 'projections': (vertices, n_neighbors, 2)
        projections = logarithmic_map(lrfs, neighborhoods)

        # 4.) Compute barycentric coordinates
        # 'closest_proj': (vertices, 3, n_radial, n_angular)
        closest_proj = tf.argsort(
            tf.linalg.norm(self.template - tf.expand_dims(tf.expand_dims(projections, axis=2), axis=2), axis=-1), axis=1
        )[:, :3, :, :]

        # 'projections': (vertices, 3, n_radial, n_angular, 2)
        projections = tf.gather(projections, closest_proj, batch_dims=1)

        v0 = projections[:, 2] - projections[:, 0]
        v1 = projections[:, 1] - projections[:, 0]
        v2 = self.template - projections[:, 0]

        dot00 = tf.einsum("vrai,vrai->vra", v0, v0)
        dot01 = tf.einsum("vrai,vrai->vra", v0, v1)
        dot02 = tf.einsum("vrai,vrai->vra", v0, v2)
        dot11 = tf.einsum("vrai,vrai->vra", v1, v1)
        dot12 = tf.einsum("vrai,vrai->vra", v1, v2)

        denominator = dot00 * dot11 - dot01 * dot01

        point_2_weight = (dot11 * dot02 - dot01 * dot12) / denominator
        point_1_weight = (dot00 * dot12 - dot01 * dot02) / denominator
        point_0_weight = 1 - point_2_weight - point_1_weight

        interpolation_weights = tf.stack([point_2_weight, point_1_weight, point_0_weight], axis=-1)

        # 5.) Get projection indices
        # 'projections_indices': (vertices, 3, n_radial, n_angular)
        projections_indices = tf.cast(tf.gather(neighborhoods_indices, closest_proj, batch_dims=1), tf.float32)

        # 6.) Return barycentric coordinates tensor
        # (vertices, n_radial, n_angular, 3, 2)
        return tf.stack([tf.transpose(projections_indices, perm=[0, 2, 3, 1]), interpolation_weights], axis=-1)
