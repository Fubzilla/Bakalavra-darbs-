import cv2
import os
import numpy as np
import open3d as o3d
import itertools
from FeatureExtractAndMatching import features, matching, gray

def fakeCameraSetup(width, height, f=1000):
    return np.array([
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ])

def circularSFM(n, radius=1.0):
    poses = []
    positions = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        cam_pos = np.array([radius * np.cos(theta), 0, radius * np.sin(theta)])
        positions.append(cam_pos)

        z = -cam_pos / np.linalg.norm(cam_pos)
        y = np.array([0, 1, 0])
        x = np.cross(y, z)
        y = np.cross(z, x)

        R = np.stack((x, y, z), axis=1) 
        t = -R @ cam_pos.reshape(-1, 1)

        poses.append((R, t))
    return poses, positions

def triangulate(K, R1, t1, R2, t2, pts1, pts2):
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = (pts4D[:3] / pts4D[3]).T
    return pts3D

def open3dForm(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def saveCamerasCircular(positions, filename):
    geometries = []
    points = []
    colors = []

    for idx, pos in enumerate(positions):
        points.append(pos)
        colors.append([0, 0, 1]) 

    points.append([0, 0, 0])
    colors.append([1, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {filename}")

def main():
    gray_dir = "output/gray"
    gray_imgs = gray(gray_dir)
    if not gray_imgs:
        print("no images")
        return

    h, w = gray_imgs[0].shape[:2]
    K = fakeCameraSetup(w, h)
    poses, positions = circularSFM(len(gray_imgs))

    saveCamerasCircular(positions, "output/sparse_circular.ply")

    keypoints, descriptors = features(gray_imgs)
    _, GoodPoints = matching(keypoints, descriptors, gray_imgs)

    PointsList = []
    for (i, j), (pts1, pts2) in GoodPoints.items():
        if len(pts1) < 8:
            continue

        R1, t1 = poses[i]
        R2, t2 = poses[j]
        pts3D = triangulate(K, R1, t1, R2, t2, pts1, pts2)
        PointsList.append(pts3D)
        print(f"Triangulated {len(pts3D)} points {i}---{j}")

    if PointsList:
        merged = np.vstack(PointsList)
        pcd = open3dForm(merged)
        
        o3d.visualization.draw_geometries([pcd])
    else:
        print("Unlucky")

if __name__ == "__main__":
    main()
