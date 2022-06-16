from GeodesicPolarMap.discrete_gpc import compute_vector_angle
import numpy as np
import c_extension

if __name__ == "__main__":

    # Example usage of the C-extension

    result = np.array([0., 0.])
    vertex_i = np.array([0.09, 0.539, 0.3076])
    vertex_j = np.array([0.10, 0.52, 0.3079])
    vertex_k = np.array([0.11, 0.537, 0.3])
    u_j = 0.01
    u_k = 0.02
    theta_j = 1.93
    theta_k = 3.55

    c_extension.compute_dist_and_dir(
        result,
        vertex_i,
        vertex_j,
        vertex_k,
        u_j,
        u_k,
        theta_j,
        theta_k
    )
    print(result)

    angle_1 = c_extension.compute_angle(vertex_i, vertex_j)
    angle_2 = compute_vector_angle(vertex_i, vertex_j)
    print(angle_1, angle_2)
