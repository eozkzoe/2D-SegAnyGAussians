import json
import numpy as np
import open3d as o3d


def load_points_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    points = np.array([item["point"] for item in data])
    normals = np.array([item["normal"] for item in data])
    return points, normals


def create_point_cloud(points, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


if __name__ == "__main__":
    source_points, source_normals = load_points_from_json("source.json")
    target_points, target_normals = load_points_from_json("target.json")
    source_pcd = create_point_cloud(source_points, source_normals)
    target_pcd = create_point_cloud(target_points, target_normals)
    threshold = 0.02  # max correspondence points-pair distance
    trans_init = np.eye(4)  # initial alignment

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    print("Transformation Matrix:")
    print(reg_p2p.transformation)

    source_pcd.transform(reg_p2p.transformation)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    o3d.visualization.draw_geometries(
        [
            source_pcd.paint_uniform_color([1, 0, 0]),
            target_pcd.paint_uniform_color([0, 1, 0]),
        ]
    )
