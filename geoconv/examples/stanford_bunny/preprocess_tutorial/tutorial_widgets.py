from geoconv.preprocessing.barycentric_coordinates import create_template_matrix

from PIL import Image
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt

from geoconv.utils.visualization import draw_gpc_triangles


def check_gpc_systems(image_dir, all_alphas, alpha=0):
    # Load pre-computed images
    radial_coordinates_3d = Image.open(f"{image_dir}/bunny_gpc_system_{alpha}_radial_coords.png")
    angular_coordinates_3d = Image.open(f"{image_dir}/bunny_gpc_system_{alpha}_angular_coords.png")
    gpc_system_2d = Image.open(f"{image_dir}/2d_gpc_system_{alpha}.png")

    # Configure plot
    fig = plt.figure(layout="constrained", figsize=(15, 11))

    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(radial_coordinates_3d)
    # ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(angular_coordinates_3d)
    # ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, :])
    ax3.imshow(gpc_system_2d)
    ax3.axis("off")

    fig.suptitle(f"Local GPC-system with 'alpha = {all_alphas[alpha]:.3f}'")
    plt.show()


def summary_visualization(normalized_bunny, idx, r, vertex_idx=0, N_rho=5, N_theta=8, alpha=.1):
    R = r * alpha
    template_vertices = create_template_matrix(n_radial=N_rho, n_angular=N_theta, radius=R * 0.75, in_cart=True)
    one_gpc_system = draw_gpc_triangles(
        normalized_bunny,
        vertex_idx,
        u_max=R,
        template_matrix=template_vertices,
        print_scatter=False,  # Put dots at the triangle vertices
        plot=True,  # Plot the image
        title="Template in GPC-system",  # Title of the Image
        save_name=f"./2d_gpc_system_{idx}.svg"  # If a `save_name` is given, the image will be saved. Otherwise it wont.
    )
