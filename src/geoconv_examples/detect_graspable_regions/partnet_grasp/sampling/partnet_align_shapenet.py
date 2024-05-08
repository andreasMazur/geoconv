from scipy.spatial.distance import cdist

from geoconv_examples.detect_graspable_regions.partnet_grasp.sampling.partnet_grasp_meshes import ANNOT_DICT

import os
import os.path as osp
import trimesh
import json
import numpy as np
import open3d as o3d
import urllib.request


# https://github.com/daerduoCarey/partnet_seg_exps/tree/master/stats/train_val_test_split
PARTNET_EXP_GITHUB_LINK = (
    "https://raw.githubusercontent.com/daerduoCarey/partnet_seg_exps/master/stats/train_val_test_split/"
)


"""(Incomplete) mapping from wordnet synsets to ShapeNet categories"""
CAT2SYNSET = dict(
    mug="03797390",
    bag="02773838",
    bottle="02876657",
    can="02946921",
    vessel="04530566",
)


LABLEMAP = dict(
    body=1,
    handle=2,
    containing_things=4,
    other=4,
)


TMPMAP = {
    0.: 'unknown',
    1.: 'body',
    2.: 'handle',
    4.: 'containing_things',
    3.: 'unknown',
    5.: 'unknown'
}

COLORMAP = dict(
    unknown=[255, 0, 0, 255],
    handle=[0, 255, 0, 255],
    body=[255, 0, 0, 255],
    containing_things=[213, 200, 48, 255],
    other=[213, 200, 48, 255],
)


def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))
    return np.vstack(vertices), np.vstack(faces)


def get_recurrent_objs(base_partnet_path, label, anno_id, to_parse):
    out_dict = {}
    if 'children' in to_parse.keys():
        for child in to_parse['children']:
            res = get_recurrent_objs(base_partnet_path, label, anno_id, child)
            out_dict = {**out_dict, **res}
    else:
        for model in to_parse['objs']:
            cur_vs, cur_fs = load_obj(osp.join(base_partnet_path, anno_id, 'objs', f'{model}.obj'))
            o3d_mesh = trimesh.Trimesh(vertices=cur_vs, faces=cur_fs-1).as_open3d
            o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
            out_dict[o3d_mesh] = label
    return out_dict


def rot_mat(angle):
    return np.array([
        np.cos(angle), 0, np.sin(angle), 0,
        0, 1, 0, 0,
        -np.sin(angle), 0, np.cos(angle), 0,
        0, 0, 0, 1
    ]).reshape(4, 4)


def align_to_partnet(
        base_partnet_path: str,
        base_shapenet_path: str,
        anno_id: str,
        cat_name: str,
        model_id: str,
        aligned_shapenet_target_path: str = None,
        verbose: bool = False,
        show_meshes: bool = False
    ) -> None:
    """Saves a ShapeNet mesh that was heuristically aligned to the corresponding PartNet mesh for further processing.

    Args:
        base_partnet_path (str): Path to root PartNet folder
        base_shapenet_path (str): Path to root ShapeNet folder
        anno_id (str): id of the model in PartNet
        cat_name (str): category the model belongs to
        model_id (str): hash (id) of the ShapeNet mesh
        aligned_shapenet_target_path (str, optional): Root path to where the mesh should be saved to.
         Defaults to ./PartNetAligned.
        verbose (bool, optional): Whether to print model statistics after transformations were applied.
         Defaults to False.
        show_meshes (bool, optional): Whether overlayed meshes should be displayed for inspection purposes.
         Defaults to False.

    Raises:
        ValueError: If the category is unknown
        ValueError: If the ShapeNet directory does not exist
        ValueError: If the ShapeNet model could not be loaded
    """

    if aligned_shapenet_target_path is None:
        aligned_shapenet_target_path = './PartNetAligned'

    os.makedirs(aligned_shapenet_target_path, exist_ok=True)
    input_objs_dir = osp.join(base_partnet_path, anno_id, 'objs')

    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

    def shapenet_trans(verts: np.ndarray) -> np.ndarray:
        """Computes the transform from the given mesh to a unit xy-diagonal and axis centered mesh
           and returns the inverted transform.

        Args:
            verts (np.ndarray): Vertices of the mesh to get the transform to.

        Returns:
            np.ndarray: Homogeneus transformation matrix from a unit xy-diagonal and centered mesh to the given mesh.
        """
        x_min = np.min(verts[:, 0])
        x_max = np.max(verts[:, 0])
        x_center = (x_min + x_max) / 2
        x_len = x_max - x_min
        y_min = np.min(verts[:, 1])
        y_max = np.max(verts[:, 1])
        y_center = (y_min + y_max) / 2
        y_len = y_max - y_min
        z_min = np.min(verts[:, 2])
        z_max = np.max(verts[:, 2])
        z_center = (z_min + z_max) / 2
        z_len = z_max - z_min

        scale = np.sqrt(x_len**2 + y_len**2)
        # scale = max(x_len, max(y_len, z_len))
        # scale = 1/max_dim

        trans = np.array([
            [0, 0, 1.0 / scale, -x_center/scale],
            [0, 1.0 / scale, 0, -y_center/scale],
            [-1/scale, 0, 0, -z_center/scale],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        trans = np.linalg.inv(trans)
        return trans

    def transform_verts(trans: np.ndarray,
                        in_verts: np.ndarray,
                        in_faces: np.ndarray,
                        rot: np.ndarray = None) -> trimesh.Trimesh:
        """Applies homogeneous transformations to a given set of vertices and returns the resulting mesh.

        Args:
            trans (np.ndarray): Homogeneus transformation matrix.
            in_verts (np.ndarray): Vertices of the mesh to transform.
            in_faces (np.ndarray): Faces of the mesh to transform.
            rot (np.ndarray, optional): Additional rotation to apply to the mesh. Defaults to None.

        Returns:
            trimesh.Trimesh: Transformed mesh.
        """
        verts = np.array(in_verts, dtype=np.float32)
        verts = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1)
        if rot is not None:
            verts = verts @ rot @ (trans.T)
        else:
            verts = verts @ (trans.T)
        verts = verts[:, :3]

        out_mesh = trimesh.Trimesh(vertices=verts, faces=in_faces)
        return out_mesh

    vs = []
    fs = []
    vid = 0
    for item in os.listdir(input_objs_dir):
        if item.endswith('.obj'):
            cur_vs, cur_fs = load_obj(osp.join(input_objs_dir, item))
            vs.append(cur_vs)
            fs.append(cur_fs + vid)
            vid += cur_vs.shape[0]

    v_arr = np.concatenate(vs, axis=0)
    v_arr_ori = np.array(v_arr, dtype=np.float32)
    f_arr = np.concatenate(fs, axis=0)
    tmp = np.array(v_arr[:, 0], dtype=np.float32)
    v_arr[:, 0] = v_arr[:, 2]
    v_arr[:, 2] = -tmp
    trans = shapenet_trans(v_arr)

    shapenet_dir = osp.join(base_shapenet_path, CAT2SYNSET[cat_name], model_id)
    out_file = osp.join(aligned_shapenet_target_path, CAT2SYNSET[cat_name], model_id, 'training', 'model_normalized.obj')
    os.makedirs(osp.join(aligned_shapenet_target_path, CAT2SYNSET[cat_name], model_id, 'training'), exist_ok=True)
    if not osp.exists(shapenet_dir):
        raise ValueError(f"Shapenet dir {shapenet_dir} does not exist!")
    tmp_mesh = trimesh.load(osp.join(shapenet_dir, 'training', 'model_normalized.obj'))

    if isinstance(tmp_mesh, trimesh.Scene):
        shapenet_mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in tmp_mesh.geometry.values())
        )
    elif isinstance(tmp_mesh, trimesh.Trimesh):
        shapenet_mesh = trimesh.Trimesh(vertices=tmp_mesh.vertices, faces=tmp_mesh.faces)
    else:
        raise ValueError("ERROR: failed to correctly load shapenet mesh!")

    # Test dist
    partnet_mesh = trimesh.Trimesh(vertices=v_arr_ori, faces=f_arr-1)
    partnet_pts = trimesh.sample.sample_surface(partnet_mesh, 2000)[0]

    shapenet_mesh_t = transform_verts(trans, shapenet_mesh.vertices, shapenet_mesh.faces)
    shapenet_pts = trimesh.sample.sample_surface(shapenet_mesh_t, 2000)[0]

    dist_mat = cdist(shapenet_pts, partnet_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()

    if chamfer_dist > 0.1:
        print(f"Misalignment detected ({chamfer_dist}), trying rotation")
        shapenet_mesh_t = transform_verts(trans, shapenet_mesh.vertices, shapenet_mesh.faces, rot_mat(np.pi/2))
        shapenet_pts  = trimesh.sample.sample_surface(shapenet_mesh_t, 2000)[0]
        dist_mat = cdist(shapenet_pts, partnet_pts)
        chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
    print(f"Chamfer Distance: {chamfer_dist}")

    # Align centers 
    verts = partnet_mesh.vertices
    p_x_min = np.min(verts[:, 0])
    p_x_max = np.max(verts[:, 0])
    p_x_len = p_x_max - p_x_min
    p_y_min = np.min(verts[:, 1])
    p_y_max = np.max(verts[:, 1])
    p_y_len = p_y_max - p_y_min
    p_z_min = np.min(verts[:, 2])
    p_z_max = np.max(verts[:, 2])
    p_z_len = p_z_max - p_z_min

    verts = shapenet_mesh_t.vertices
    s_x_min = np.min(verts[:, 0])
    s_x_max = np.max(verts[:, 0])
    s_x_len = s_x_max - s_x_min
    s_y_min = np.min(verts[:, 1])
    s_y_max = np.max(verts[:, 1])
    s_y_len = s_y_max - s_y_min
    s_z_min = np.min(verts[:, 2])
    s_z_max = np.max(verts[:, 2])
    s_z_len = s_z_max - s_z_min

    f = p_x_len / s_x_len
    scale = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, f, 0],
        [0, 0, 0, 0]
    ])

    shapenet_mesh_t = transform_verts(scale, shapenet_mesh_t.vertices, shapenet_mesh_t.faces)
    shapenet_pts  = trimesh.sample.sample_surface(shapenet_mesh_t, 2000)[0]
    dist_mat = cdist(shapenet_pts, partnet_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
    print(f"Chamfer Distance new scale: {chamfer_dist}")

    verts = shapenet_mesh_t.vertices
    s_x_min = np.min(verts[:, 0])
    s_x_max = np.max(verts[:, 0])
    s_x_len = s_x_max - s_x_min
    s_y_min = np.min(verts[:, 1])
    s_y_max = np.max(verts[:, 1])
    s_y_len = s_y_max - s_y_min
    s_z_min = np.min(verts[:, 2])
    s_z_max = np.max(verts[:, 2])
    s_z_len = s_z_max - s_z_min

    offset = np.array([
        [1, 0, 0, p_x_min - s_x_min],
        [0, 1, 0, p_y_min - s_y_min],
        [0, 0, 1, p_z_min - s_z_min],
        [0, 0, 0, 0]
    ])
    shapenet_mesh_t = transform_verts(offset, shapenet_mesh_t.vertices, shapenet_mesh_t.faces)
    shapenet_pts = trimesh.sample.sample_surface(shapenet_mesh_t, 2000)[0]
    dist_mat = cdist(shapenet_pts, partnet_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
    print(f"Chamfer Distance new t: {chamfer_dist}")

    verts = shapenet_mesh_t.vertices
    s_x_min = np.min(verts[:, 0])
    s_x_max = np.max(verts[:, 0])
    s_x_len = s_x_max - s_x_min
    s_y_min = np.min(verts[:, 1])
    s_y_max = np.max(verts[:, 1])
    s_y_len = s_y_max - s_y_min
    s_z_min = np.min(verts[:, 2])
    s_z_max = np.max(verts[:, 2])
    s_z_len = s_z_max - s_z_min

    if verbose:
        print(f"partnet measures:\n{p_x_len}, {p_y_len}, {p_z_len}; {p_z_len**2+p_y_len**2}")
        print(f"shapenet measures:\n{s_x_len}, {s_y_len}, {s_z_len}; {s_y_len**2+s_z_len**2}")

    trimesh.exchange.export.export_mesh(shapenet_mesh_t, out_file, file_type='obj')

    if show_meshes:
        mesh = trimesh.util.concatenate([partnet_mesh, shapenet_mesh_t])
        colors = np.concatenate(
            [
                np.repeat([[0, 255, 0, 255]], len(partnet_mesh.vertices)).reshape(4, -1).T,
                np.repeat([[255, 127, 0, 255]], len(shapenet_mesh_t.vertices)).reshape(4, -1).T
            ]
        )
        print(colors.shape, len(mesh.vertices))
        pc = trimesh.PointCloud(vertices=mesh.vertices, colors=colors)
        pc.show()


class PartNetAnnotatedMesh:
    def __init__(self, anno_id: str, base_partnet_path: str, base_shapenet_path: str):
        self.anno_id = anno_id
        self.base_PartNet_path = base_partnet_path
        self.base_ShapeNet_path = base_shapenet_path

        with open(f"{base_partnet_path}/{anno_id}/result.json") as f:
            self.result = json.load(f)[0]
            self.segments = self.result['children']

        with open(f"{base_partnet_path}/{anno_id}/meta.json") as f:
            self.meta = json.load(f)

        self.objs = os.listdir(f"{base_partnet_path}/{anno_id}/objs")

        cat_id = CAT2SYNSET[self.meta['model_cat'].lower()]
        self.ShapeNet_model_path = (
            f"{self.base_ShapeNet_path}/{cat_id}/{self.meta['model_id']}/training/model_normalized.obj"
        )

    def print_paths(self):
        print(self.ShapeNet_model_path)
        print(f"{self.base_PartNet_path}/{self.anno_id}/objs/{self.objs[0]}")


USER_COLORS = [
    [0, 255, 0, 255],
    [255, 127, 0, 255],
    [119, 221, 231, 255],
]


def convert_partnet_labels(
        base_partnet_path: str,
        base_shapenet_path: str,
        target_mesh_path: str,
        target_dataset_path: str,
        obj_class: str = 'Mug',
        manual: bool = False,
        verbose: bool = False) -> None:
    """Transfers PartNet labels to ShapeNet mesh.
    Involves
    a) extracting ShapeNet meshes of the given class from ShapeNet,
    b) transforming the ShapeNet mesh to match the combined PartNet transform,
    c) assigning labels to the transformed ShapeNet mesh by choosing the min distance of signed distance with each
     individual segment.
    d) saving vertices, faces, and vertex labels as npz files under `target_dataset_path`.

    Args:
        base_partnet_path (str): Path to `PartNet/data_vX` dir.
        base_shapenet_path (str): Path to `ShapeNetCore.vX` dir.
        target_mesh_path (str): Destination dir to save the aligned ShapeNet partnet_grasp to.
        target_dataset_path (str): Destination dir to save the extracted dataset to.
        obj_class (str, optional): Object class name. Defaults to 'Mug'.
        manual (bool, optional): Whether to manually select meshes. Defaults to False.
        verbose (bool, optional): Show more info. Defaults to False.
    """

    data = []
    for split in ['train', 'test', 'val']:
        with urllib.request.urlopen(f"{PARTNET_EXP_GITHUB_LINK}/{obj_class}.{split}.json") as url:
            d = json.load(url)
            data.extend(d)

    for inst in data:
        annot = PartNetAnnotatedMesh(
            anno_id=inst["anno_id"],
            base_partnet_path=base_partnet_path,
            base_shapenet_path=base_shapenet_path
        )

        align_to_partnet(
            base_partnet_path=base_partnet_path,
            base_shapenet_path=base_shapenet_path,
            aligned_shapenet_target_path=target_mesh_path,
            anno_id=annot.anno_id,
            cat_name=annot.meta['model_cat'].lower(),
            model_id=annot.meta['model_id'],
            verbose=verbose
        )

        use_mesh = None
        if manual:
            use_mesh = manual_filter_mesh(
                base_partnet_path=base_partnet_path,
                aligned_shapenet_path=target_mesh_path,
                anno_id=annot.meta['anno_id'],
                cat_name=annot.meta['model_cat'].lower(),
                model_id=annot.meta['model_id']
            )
        else:
            use_mesh = ANNOT_DICT[inst['anno_id']]

        if use_mesh:
            orig_segmentation(
                base_partnet_path=base_partnet_path,
                aligned_shapenet_path=target_mesh_path,
                annot=annot,
                out_npz_dir=target_dataset_path,
                verbose=verbose,
                manual=manual
            )


def manual_filter_mesh(
        base_partnet_path: str,
        aligned_shapenet_path: str,
        anno_id: str,
        cat_name: str,
        model_id: str) -> bool:

    input_objs_dir = osp.join(base_partnet_path, anno_id, 'objs')

    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

    vs = []
    fs = []
    vid = 0
    for item in os.listdir(input_objs_dir):
        if item.endswith('.obj'):
            cur_vs, cur_fs = load_obj(osp.join(input_objs_dir, item))
            vs.append(cur_vs)
            fs.append(cur_fs + vid)
            vid += cur_vs.shape[0]

    v_arr = np.concatenate(vs, axis=0)
    v_arr_ori = np.array(v_arr, dtype=np.float32)
    f_arr = np.concatenate(fs, axis=0)
    tmp = np.array(v_arr[:, 0], dtype=np.float32)
    v_arr[:, 0] = v_arr[:, 2]
    v_arr[:, 2] = -tmp

    shapenet_dir = osp.join(aligned_shapenet_path, CAT2SYNSET[cat_name], model_id)
    if not osp.exists(shapenet_dir):
        raise ValueError(f"Shapenet dir {shapenet_dir} does not exist!")
    tmp_mesh = trimesh.load(osp.join(shapenet_dir, 'training', 'model_normalized.obj'))

    if isinstance(tmp_mesh, trimesh.Scene):
        shapenet_mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in tmp_mesh.geometry.values())
        )
    elif isinstance(tmp_mesh, trimesh.Trimesh):
        shapenet_mesh = trimesh.Trimesh(vertices=tmp_mesh.vertices, faces=tmp_mesh.faces)
    else:
        raise ValueError("ERROR: failed to correctly load shapenet mesh!")

    # Test dist
    partnet_mesh = trimesh.Trimesh(vertices=v_arr_ori, faces=f_arr-1)
    partnet_pts = trimesh.sample.sample_surface(partnet_mesh, 2000)[0]

    shapenet_pts = trimesh.sample.sample_surface(shapenet_mesh, 2000)[0]

    dist_mat = cdist(shapenet_pts, partnet_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()

    print(f"Chamfer Distance: {chamfer_dist}")

    mesh = trimesh.util.concatenate([partnet_mesh, shapenet_mesh])
    colors = np.concatenate(
        [
            np.repeat([[0, 255, 0, 255]], len(partnet_mesh.vertices)).reshape(4, -1).T,
            np.repeat([[255, 127, 0, 255]], len(shapenet_mesh.vertices)).reshape(4, -1).T
        ]
    )
    print(colors.shape, len(mesh.vertices))
    pc = trimesh.PointCloud(vertices=mesh.vertices, colors=colors)
    pc.show()

    ans = ''
    valid = {'yes': True, 'y': True, 'no': False, 'n': False}
    while ans not in valid.keys():
        ans = input("Use mesh? [y/n]: ")
    return valid[ans]


def orig_segmentation(
            base_partnet_path: str,
            aligned_shapenet_path: str,
            annot: PartNetAnnotatedMesh,
            out_npz_dir: str,
            verbose: bool = False,
            manual: bool = False
        ) -> None:
    """Computes aligned ShapeNet labels from signed distance to PartNet segments.

    Args:
        base_partnet_path (str): Path to `PartNet/data_vX` dir.
        aligned_shapenet_path (str): Path to aligned ShapeNet partnet_grasp root dir.
        annot (PartNetAnnotatedMesh): Mesh to process.
        out_npz_dir (str): Directory where to save a numpy file containing vertices, faces, and segment labels.
        verbose (bool, optional): Print more info. Defaults to False.
        TODO -- manual docstring

    Raises:
        ValueError: If the category is unknown
        ValueError: If the aligned ShapeNet directory does not exist
        ValueError: If the aligned ShapeNet model could not be loaded
    """

    os.makedirs(out_npz_dir, exist_ok=True)

    anno_id = annot.anno_id
    cat_name = annot.meta['model_cat'].lower()
    model_id = annot.meta['model_id']

    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

    # Load segments as individual meshes
    segments = {}
    for segment in annot.segments:
        try:
            label = LABLEMAP[segment['name']]
        except KeyError:
            print(f"ERROR: Did contain segment class {segment['name']}. Assigning 'other'")
            label = 4

        if verbose:
            print(f"label is: {label} (segment: {segment})")
        tmp = get_recurrent_objs(
            base_partnet_path=base_partnet_path,
            label=label,
            anno_id=anno_id,
            to_parse=segment
        )
        segments = {**segments, **tmp}

    if verbose:
        for m in segments.keys():
            print(f"Mesh type: {type(m)}")

    shapenet_dir = osp.join(aligned_shapenet_path, CAT2SYNSET[cat_name], model_id)
    if not osp.exists(shapenet_dir):
        raise ValueError(f"Shapenet dir {shapenet_dir} does not exist!")
    tmp_mesh = trimesh.load(osp.join(shapenet_dir, 'training', 'model_normalized.obj'))

    if isinstance(tmp_mesh, trimesh.Scene):
        shapenet_mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in tmp_mesh.geometry.values())
        )
    elif isinstance(tmp_mesh, trimesh.Trimesh):
        shapenet_mesh = trimesh.Trimesh(vertices=tmp_mesh.vertices, faces=tmp_mesh.faces)
    else:
        raise ValueError("ERROR: failed to correctly load shapenet mesh!")

    res = np.zeros((shapenet_mesh.vertices.shape[0],), dtype=np.float32)
    # sadly no cuda support for embree yet , device=o3d.core.Device("CUDA:0"))
    verts = o3d.core.Tensor(shapenet_mesh.vertices, dtype=o3d.core.Dtype.Float32)
    distances = []
    lbls = []
    for mesh, lbl in segments.items():
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        dist = scene.compute_signed_distance(verts)
        distances.append(dist)
        lbls.append(lbl)

    # Get index i.e label with min dist
    res = np.stack(distances, axis=-1).argmin(axis=1)

    labels = []
    colors = []
    for _, vert in enumerate(res):
        colors.append(COLORMAP[TMPMAP[lbls[vert]]])
        labels.append(lbls[vert]-1)

    m_valid = True
    print(f"INFO: vertcount: {verts.shape[0]}")
    if verts.shape[0] > 10000:
        print(f"WARNING: vertices count over 10k ({verts.shape[0]}). Discarding mesh")
        m_valid = False
    elif (np.vstack(lbls) > 2).any():
        classes = [segment['name'] for segment in annot.segments]
        print(f"WARNING: has content segment in mesh ({classes}). Discarding mesh")
        m_valid = False

    ans = ''
    if not m_valid:
        ans = 'no'
    elif not manual:
        ans = 'yes'
    else:
        colors = np.vstack(colors)
        pc = trimesh.PointCloud(vertices=shapenet_mesh.vertices, colors=colors)
        pc.show()

    valid = {'yes': True, 'y': True, 'no': False, 'n': False}

    while ans not in valid.keys():
        ans = input("Mesh valid? [y/n]: ")
    if valid[ans]:
        labels = np.vstack(labels)
        with open(f'{out_npz_dir}/{annot.anno_id}.npz', 'wb') as f:
            np.savez(f, verts=shapenet_mesh.vertices, faces=shapenet_mesh.faces, labels=labels)
