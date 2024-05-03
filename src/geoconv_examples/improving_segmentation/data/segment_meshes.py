from geoconv_examples.improving_segmentation.data.bounding_box import BoundingBox
from geoconv_examples.improving_segmentation.data.bounding_box_utils import visualize_bbs

import numpy as np
import trimesh


MESH_SEGMENTS = {
    0: "right_arm",
    1: "left_arm",
    2: "torso",
    3: "head",
    4: "left_foot",
    5: "right_foot",
    6: "left_leg",
    7: "right_leg",
    8: "left_hand",
    9: "right_hand"
}


def segment_mesh(mesh, bb_configurations, verbose=False):
    """Assigns mesh vertices to segments by checking whether they are included in a bounding box.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The mesh which shall be segmented.
    bb_configurations: list
        A list containing dictionaries containing bounding box descriptions for segment definition:
            [
                {"segment_1":
                    [
                        ((x_1_1, y_1_1, z_1_1), width_1_1, height_1_1, depth_1_1),  # Bounding box description
                        ((x_1_2, y_1_2, z_1_2), width_1_2, height_1_2, depth_1_2)
                    ]
                },
                {"segment_2": [((x_2_1, y_2_1, z_2_1), width_2_1, height_2_1, depth_2_1)]}
            ]
    verbose: bool
        Whether to plot the segments and bounding boxes.

    Returns
    -------
    dict:
        A dictionary describing which vertex belongs to which segments. Vertices may appear in multiple segments.
    """
    vertex_segments_relation = {}
    for idx, bb_conf in enumerate(bb_configurations):
        bbs = [
            BoundingBox(anchor, width, height, depth) for (anchor, width, height, depth) in list(bb_conf.values())[0]
        ]
        included_vertices = visualize_bbs(mesh, bbs, verbose)
        vertex_segments_relation[list(bb_conf.keys())[0]] = included_vertices
    return vertex_segments_relation


def compute_seg_labels(registration_path, verbose=False):
    """Determine the vertex labels in the context of FAUST segmentation.

    The FAUST-data set has to be downloaded from: https://faust-leaderboard.is.tuebingen.mpg.de/

    It was published in:
    > [FAUST: Dataset and evaluation for 3D mesh registration.]
    (https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Bogo_FAUST_Dataset_and_2014_CVPR_paper.html)
    > Bogo, Federica, et al.

    Parameters
    ----------
    registration_path: str
        The path to the registrations folder of the FAUST dataset.
    verbose: bool
        Whether to plot the segments.

    Returns
    -------
    dict:
        A dictionary describing which vertex belongs to which segments. Vertices may appear in multiple segments.
    """

    ###################################################
    # Configure segments by configuring bounding boxes
    ###################################################
    mesh = trimesh.load_mesh(f"{registration_path}/tr_reg_009.ply")
    vertex_segments_relation = segment_mesh(
        mesh,
        # [(anchor_point, width, height, depth), ...] <- One list per segment
        [
            {"right_arm": [((-.35, 0.4, -0.05), 0.25, 0.42, 0.35), ((-.28, 0.3, -0.025), 0.225, 0.12, 0.15)]},
            {"left_arm": [
                ((-.35 + 0.6, 0.4, -0.05), 0.25, 0.42 - 0.04, 0.35), ((-.3 + 0.44, 0.3, -0.01), 0.225, 0.13, 0.17)
            ]},
            {"torso": [
                ((-.17, -.3, -.05), 0.45, 0.67, 0.3), ((-0.06, .35, -.05), 0.2, 0.037, 0.15),
            ]},
            {"head": [((-.06, .37, 0.03), 0.21, 0.35, 0.28)]},
            {"left_leg": [((mesh.vertices.mean(axis=0)[0] - 0.03, -1.2, -.1), 0.25, 0.9, 0.35)]},
            {"right_leg": [((-.24, -1.2, -.1), 0.26, 0.9, 0.35)]},
            {"left_hand": [((.325, 0.78, 0.1), 0.2, 0.225, 0.25)]},
            {"right_hand": [((-.4, 0.82, 0.1), 0.22, 0.225, 0.25)]},
            {"right_foot": [((-.22, -1.2, -.1), 0.25, 0.2, 0.35)]},
            {"left_foot": [((mesh.vertices.mean(axis=0)[0], -1.2, -.1), 0.25, 0.2, 0.35)]}
        ],
        verbose=verbose
    )

    #############################
    # Vertex ground truth labels
    #############################
    classes = []
    for idx, _ in enumerate(mesh.vertices):
        if idx in vertex_segments_relation["right_arm"]:
            classes.append(0)
        elif idx in vertex_segments_relation["left_arm"]:
            classes.append(1)
        elif idx in vertex_segments_relation["torso"]:
            classes.append(2)
        elif idx in vertex_segments_relation["head"]:
            classes.append(3)
        elif idx in vertex_segments_relation["left_foot"]:
            classes.append(4)
        elif idx in vertex_segments_relation["right_foot"]:
            classes.append(5)
        elif idx in vertex_segments_relation["left_leg"]:
            classes.append(6)
        elif idx in vertex_segments_relation["right_leg"]:
            classes.append(7)
        elif idx in vertex_segments_relation["left_hand"]:
            classes.append(8)
        elif idx in vertex_segments_relation["right_hand"]:
            classes.append(9)
        else:
            raise RuntimeError(f"Vertex {idx} has no class!")

    if verbose:
        ################
        # Vertex colors
        ################
        colors = []
        for idx, _ in enumerate(mesh.vertices):
            if idx in vertex_segments_relation["right_arm"]:
                colors.append([255, 0, 0, 255])
            elif idx in vertex_segments_relation["left_arm"]:
                colors.append([0, 255, 0, 255])
            elif idx in vertex_segments_relation["torso"]:
                colors.append([0, 0, 255, 255])
            elif idx in vertex_segments_relation["head"]:
                colors.append([255, 255, 0, 255])
            elif idx in vertex_segments_relation["left_foot"]:
                colors.append([100, 255, 0, 255])
            elif idx in vertex_segments_relation["right_foot"]:
                colors.append([100, 0, 255, 255])
            elif idx in vertex_segments_relation["left_leg"]:
                colors.append([255, 0, 255, 255])
            elif idx in vertex_segments_relation["right_leg"]:
                colors.append([0, 255, 255, 255])
            elif idx in vertex_segments_relation["left_hand"]:
                colors.append([255, 255, 255, 255])
            elif idx in vertex_segments_relation["right_hand"]:
                colors.append([100, 0, 0, 255])
            else:
                colors.append([0, 0, 0, 255])

        file_numbers = ["".join(["0" for _ in range(3 - len(f"{i}"))]) + f"{i}" for i in range(100)]
        for fo in file_numbers[0:100:10]:
            mesh = trimesh.load_mesh(f"{registration_path}/tr_reg_{fo}.ply")
            pc = trimesh.PointCloud(vertices=mesh.vertices, colors=colors)
            pc.show()

    return np.array(classes)
