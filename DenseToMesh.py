import open3d as o3d
import numpy as np

def pointcloud(pointcloudFolder):
    pcd = o3d.io.read_point_cloud(pointcloudFolder)
    if not pcd.has_normals():
        print("Normals")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def meshBallPivo(pcd):
    print("Ball pivoting")
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    radii = [radius, radius * 2]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    return mesh

def poissonMeshRec(pcd, depth=9):
    print(f"Poisson surface {depth}")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh

def meshAlpha(pcd, alpha=0.03):
    print(f"Alpha shape surface {alpha}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh

def saving(mesh, output_path):
    print(f"Saving {output_path}")
    o3d.io.write_triangle_mesh(output_path, mesh)

def main():
    pointcloudFolder = "output/fused_icp_dpt.ply"  
    pcd = pointcloud(pointcloudFolder)

    mesh = meshBallPivo(pcd)
    

    mesh.compute_vertex_normals()
    saving(mesh, "output/mesh.ply")
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main()
