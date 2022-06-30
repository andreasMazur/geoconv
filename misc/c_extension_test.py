from preprocessing.discrete_gpc import compute_vector_angle
import numpy as np
import c_extension

if __name__ == "__main__":

    # Example usage of the C-extension
    # result = np.array([0., 0.])
    # rotation_axis = np.array([0., 0., 1.])
    # vertex_i = np.array([0.09, 0.539, 0.3076])
    # vertex_j = np.array([0.10, 0.52, 0.3079])
    # vertex_k = np.array([0.11, 0.537, 0.3])
    # u_j = 0.01
    # u_k = 0.02
    # theta_j = 1.93
    # theta_k = 3.55
    #
    # c_extension.compute_dist_and_dir(
    #     result,
    #     vertex_i,
    #     vertex_j,
    #     vertex_k,
    #     u_j,
    #     u_k,
    #     theta_j,
    #     theta_k,
    #     rotation_axis
    # )
    # print(result)

    # x_axis = np.array([1., 0., 0.])
    # y_axis = np.array([0., 1., 0.])
    # z_axis = np.array([0., 0., 1.])
    # angle_1 = c_extension.compute_angle(y_axis, x_axis)
    # py_angle_1 = compute_vector_angle(y_axis, x_axis, None)
    # print(f"Angle(y_axis, x_axis): Python {py_angle_1} - C {angle_1}")

    vertex1 = np.array([-0.00117344, -0.0023157, 0.01018211])
    vertex2 = np.array([-0.00117344, -0.0023157, 0.01018211])
    rot_axis = np.array([0.99052361, 0.13718135, 0.00665194])
    angle_2 = c_extension.compute_angle_360(vertex1, vertex2, rot_axis)
    py_angle_2 = compute_vector_angle(vertex1, vertex2, rot_axis)
    print(f"Angle_360(y_axis, x_axis): Python {py_angle_2} - C {angle_2}")
