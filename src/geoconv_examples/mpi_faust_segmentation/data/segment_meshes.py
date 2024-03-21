import trimesh
import numpy as np
import os


class BoundingBox:
    def __init__(self, anchor, width, height, depth):
        self.anchor = anchor
        self.width = width
        self.height = height
        self.depth = depth

        self.x_min = anchor[0]
        self.x_max = anchor[0] + width

        self.y_min = anchor[1]
        self.y_max = anchor[1] + height

        self.z_min = anchor[2]
        self.z_max = anchor[2] + depth

    def is_within(self, query_point):
        x_okay = self.x_min <= query_point[0] <= self.x_max
        y_okay = self.y_min <= query_point[1] <= self.y_max
        z_okay = self.z_min <= query_point[2] <= self.z_max
        return x_okay and y_okay and z_okay

    def corners(self):
        return np.array([
            [self.x_min, self.y_min, self.z_min],  # lower left
            [self.x_max, self.y_min, self.z_min],  # lower right
            [self.x_min, self.y_max, self.z_min],  # upper left
            [self.x_max, self.y_max, self.z_min],  # upper right
            [self.x_min, self.y_min, self.z_max],  # deep lower left
            [self.x_max, self.y_min, self.z_max],  # deep lower right
            [self.x_min, self.y_max, self.z_max],  # deep upper left
            [self.x_max, self.y_max, self.z_max],  # deep upper right
        ])


def visualize_bbs(mesh, bbs, verbose=False):
    """Visualizes bounding boxes.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The mesh that shall be segmented.
    bbs: list
        A list of bounding boxes.
    verbose: bool
        Whether to plot the bounding boxes

    Returns
    -------
    np.ndarray:
        Vertex indices which are within the bounding boxes.
    """
    # Initialize vertices and their colors
    vertices = np.array(mesh.vertices)
    colors = [[0, 0, 0, 255] for _ in range(vertices.shape[0])]
    are_within = []

    for bb in bbs:
        # Get bounding box corner points
        corners = bb.corners()

        # Update vertex colors depending on whether they are inside one bounding box
        for idx, vertex in enumerate(mesh.vertices):
            if bb.is_within(vertex):
                colors[idx] = [0, 255, 0, 255]
                are_within.append(idx)

        # Add corner points to plot
        vertices = np.concatenate([vertices, corners])

        # Add bounding box to colors
        colors = colors + [[255, 0, 0, 255] for _ in range(8)]

    # Visualize point cloud
    if verbose:
        pc = trimesh.PointCloud(vertices=vertices, colors=colors)
        pc.show()

    return np.unique(np.array(are_within))


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


def compute_seg_labels(registration_path, label_path, verbose=False):
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
    label_path: str
        A path in which the segmentation labels shall be saved.
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
    mesh = trimesh.load_mesh(f"{registration_path}/tr_reg_000.ply")
    vertex_segments_relation = segment_mesh(
        mesh,
        # [(anchor_point, width, height, depth), ...] <- One list per segment
        [
            {"right_arm": [((-.6, -.25, 0), 0.47, 1., 0.5), ((-.3, .07, .05), 0.2, 0.3, 0.2)]},
            {"left_arm": [((.23, -.28, -.05), 0.47, 1., 0.5), ((.219, .04, 0.01), 0.2, 0.3, 0.2)]},
            {"torso": [((-.121, -.25, -.05), 0.34, 0.6, 0.3)]},
            {"head": [((-.075, .33, 0), 0.3, 0.3, 0.35)]},
            {"left_leg": [((mesh.vertices.mean(axis=0)[0], -1.2, -.1), 0.25, 1.0, 0.35)]},
            {"right_leg": [((-.19, -1.2, -.1), 0.25, 1., 0.35)]},
            {"left_hand": [((.325, -.5, 0), 0.15, 0.225, 0.25)]},
            {"right_hand": [((-.4, -.475, 0), 0.15, 0.25, 0.25)]},
            {"right_foot": [((-.19, -1.2, -.1), 0.25, 0.2, 0.35)]},
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

    # Save labels
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    np.save(f"{label_path}/segmentation_labels.npy", classes)

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

    return vertex_segments_relation
