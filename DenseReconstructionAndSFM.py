import os
import cv2
import numpy as np
import open3d as o3d
import torch
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from PIL import Image

def midas():
    print("[midas.")
    pathToMidas = "midas/dpt_hybrid_384.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DPTDepthModel(
        path=pathToMidas,
        backbone="vitb_rn50_384",
        non_negative=True
    )
    model.to(device)
    model.eval()

    transform = Compose([
        Resize(384, 384, keep_aspect_ratio=True, ensure_multiple_of=32),
        NormalizeImage(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    return model, transform, device

def depthEstimate(img, model, transform, device):
    colorImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    maskFake = np.ones(colorImage.shape[:2], dtype=np.uint8)
    sample = {"image": colorImage, "mask": maskFake}
    sample = transform(sample)
    tensor = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    return prediction

def PointCloudFromDepthData(depth, img, K, max_depth=10.0):
    h, w = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = img.reshape(-1, 3) / 255.0

    valid = (z.reshape(-1) > 0.1) & (z.reshape(-1) < max_depth)
    return points[valid], colors[valid]

def pointCloudMake(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def FakePosesCircular(n, radius=1.5):
    poses = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        eye = np.array([radius * np.cos(angle), 0.0, radius * np.sin(angle)])
        center = np.array([0, 0, 0])
        up = np.array([0, -1, 0])

        z = (eye - center)
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        R = np.stack((x, y, z), axis=1)
        T = eye.reshape(3, 1)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T.flatten()
        poses.append(np.linalg.inv(pose))  
    return poses

def main():
    folder = "images"
    imagesNum = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".jpg")])[:6]

    model, transform, device = midas()

    pointcloudsAll = []
    sample_img = cv2.imread(imagesNum[0])
    h, w = sample_img.shape[:2]
    focal_length = 1000.0
    K = np.array([[focal_length, 0, w / 2],
                  [0, focal_length, h / 2],
                  [0, 0, 1]])

    circular_poses = FakePosesCircular(len(imagesNum))

    for idx, path in enumerate(imagesNum):
        print(f"Working on {idx+1}/{len(imagesNum)}")
        img = cv2.imread(path)
        colorImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = depthEstimate(img, model, transform, device)
        depth = cv2.medianBlur(depth.astype(np.float32), 5)
        depth = np.clip(depth, 0.1, 5.0)

        points, colors = PointCloudFromDepthData(depth, colorImage, K)

        pose = circular_poses[idx]
        hom_points = np.hstack([points, np.ones((points.shape[0], 1))])
        world_points = (pose @ hom_points.T).T[:, :3]

        pcd = pointCloudMake(world_points, colors)
        pointcloudsAll.append(pcd)

    print("ICP refine")
    merged = pointcloudsAll[0]
    for cloud in pointcloudsAll[1:]:
        reg = o3d.pipelines.registration.registration_icp(
            cloud, merged, 0.03, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        cloud.transform(reg.transformation)
        merged += cloud
        merged = merged.voxel_down_sample(voxel_size=0.003)

    os.makedirs("output", exist_ok=True)
    o3d.io.write_point_cloud("output/fused_circular_dpt.ply", merged)
    o3d.visualization.draw_geometries([merged])
    print("Done and saved")

if __name__ == "__main__":
    main()
