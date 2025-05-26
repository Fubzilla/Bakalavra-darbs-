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
    print("Midas loading")
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


def PointcloudFromDepthData(depth, img, scale=2.0):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth * scale
    x = (i - w / 2) * z / w
    y = (j - h / 2) * z / h

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = img.reshape(-1, 3) / 255.0

    print(f"[DEBUG] Z range: {z.min():.2f} to {z.max():.2f}")



    mask = (z > 0.001) & (z < 20.0)
    points = points[mask.reshape(-1)]
    colors = colors[mask.reshape(-1)]


    return points, colors

def pointCloudMake(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    folder = "images"
    imagesNum = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".jpg")])[:17]

    model, transform, device = midas()

    pointcloudsAll = []
    ReferenceC = None

    for idx, path in enumerate(imagesNum):
        print(f"Estimating depth {idx+1}/{len(imagesNum)}")
        img = cv2.imread(path)
        colorImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = depthEstimate(img, model, transform, device)

        # Normalize depth scale
        depth = depth - depth.min()
        depth = depth / depth.max()
        depth = depth * 2.0

        points, colors = PointcloudFromDepthData(depth, colorImage)
        if len(points) == 0:
            print(f"no good points {idx}")
            continue

        pcd = pointCloudMake(points, colors)

        if ReferenceC is None:
            ReferenceC = pcd
            pointcloudsAll.append(ReferenceC)
            continue

        print(f"{idx} ")
        threshold = 0.05
        reg = o3d.pipelines.registration.registration_icp(
            pcd, ReferenceC, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        aligned_pcd = pcd.transform(reg.transformation)
        pointcloudsAll.append(aligned_pcd)

    if not pointcloudsAll:
        print("Nothing was made")
        return

    print("Merging clouds")
    merged = pointcloudsAll[0]
    for cloud in pointcloudsAll[1:]:
        merged += cloud

    merged = merged.voxel_down_sample(voxel_size=0.005)
    os.makedirs("output", exist_ok=True)
    o3d.io.write_point_cloud("output/fused_icp_dpt.ply", merged)
    o3d.visualization.draw_geometries([merged])
    print("Done and saved")

if __name__ == "__main__":
    main()
