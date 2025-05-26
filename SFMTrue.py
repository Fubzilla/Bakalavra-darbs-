import numpy as np
import cv2
import os
import networkx as nx
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List, Tuple

def imagesLoaded(folder: str) -> List[np.ndarray]:
    images = []
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, file))
            if img is not None:
                images.append(img)
    return images

def matches(fileMatched: str, matchesFloor=50) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    matchesInfo = np.load(fileMatched, pickle=True)
    features = []
    for pair in matchesInfo.files:
        i, j = map(int, pair.split('_'))
        Point1, Point2 = matchesInfo[pair]
        if Point1.shape[0] >= matchesFloor:
            features.append((i, j, Point1, Point2))
    return features

def filter(features):
    G = nx.Graph()
    for i, j, Point1, Point2 in features:
        G.add_edge(i, j, weight=len(Point1))

    if not G.nodes:
        print("No matches")
        return [], G

    section = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(section).copy()

    dictionary = {(i, j): (Point1, Point2) for i, j, Point1, Point2 in features}
    node = sorted(subgraph.nodes())
    if len(node) > 2:
        first, last = node[0], node[-1]
        if (first, last) in dictionary:
            subgraph.add_edge(first, last, weight=len(dictionary[(first, last)][0]))
            print(f"closed loop {first}---{last}")
        elif (last, first) in dictionary:
            subgraph.add_edge(last, first, weight=len(dictionary[(last, first)][0]))
            print(f"closed loop {last}---{first}")
        else:
            print("cannot close loop")

    nx.draw(subgraph, with_labels=True, nodeColor='skyblue', nodeSize=800, edge='gray')
    plt.title("Closed loop")
    plt.show()

    return list(subgraph.edges()), subgraph

def fakeCamera(focal: float, width: int, height: int) -> np.ndarray:
    return np.array([[focal, 0, width / 2],
                     [0, focal, height / 2],
                     [0, 0, 1]])

def estimatedLocation(kp1, kp2, K) -> Tuple[np.ndarray, np.ndarray]:
    E, mask = cv2.findEssentialMat(kp1, kp2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, kp1, kp2, K)
    return R, t

def estimatedCameraLocation(goodPairs, features, K, numberImages, graph):
    camera = [np.eye(4) for _ in range(numberImages)]
    known = set()
    graphLocation = {}


    order = []

    seed = list(graph.nodes())[0]
    graphLocation[seed] = np.eye(4)
    known.add(seed)
    order.append(seed)

    dictionary = {(i, j): (Point1, Point2) for i, j, Point1, Point2 in features}

    while order:
        current = order.pop(0)
        for neighbor in graph.neighbors(current):
            if neighbor in known:
                continue
            if (current, neighbor) in dictionary:
                Point1, Point2 = dictionary[(current, neighbor)]
                R, t = estimatedLocation(Point1, Point2, K)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t[:, 0]
                graphLocation[neighbor] = graphLocation[current] @ np.linalg.inv(T)
            elif (neighbor, current) in dictionary:
                Point1, Point2 = dictionary[(neighbor, current)]
                R, t = estimatedLocation(Point1, Point2, K)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t[:, 0]
                graphLocation[neighbor] = graphLocation[current] @ T
            else:
                continue
            known.add(neighbor)
            order.append(neighbor)

    for idx in graphLocation:
        camera[idx] = graphLocation[idx]

    return camera

def cameraLocationShow(camera: List[np.ndarray], frame_size=0.1):
    geo = []
    for pose in camera:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(pose)
        geo.append(frame)

    o3d.visualization.draw_geometries(geo)

def saveCameraLocation(camera: List[np.ndarray], output_file: str):
    np.savez(output_file, poses=camera)
    
    print(f"save to{output_file}")

def main():
    folder = "images"
    fileMatched = "output/matches/matches.npz"
    output = "output/locations/camera.npz"
    focal = 1000.0

    images = imagesLoaded(folder)
    if not images:
        print("no image")
        return

    height, width = images[0].shape[:2]
    K = fakeCamera(focal, width, height)

    features = matches(fileMatched, matchesFloor=30)

    goodPairs, graph = filter(features)

    print(f"\n{len(graph.nodes())} views")
    print(f"edge {graph.edges()}")

    camera = estimatedCameraLocation(goodPairs, features, K, len(images), graph)

    saveCameraLocation(camera, output)
    cameraLocationShow(camera)

if __name__ == "__main__":
    main()
