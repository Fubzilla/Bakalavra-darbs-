import cv2
import os
import itertools
import numpy as np


def gray(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

def features(imgGray):
    sift = cv2.SIFT_create()
    keypoints, descriptors = [], []
    for img in imgGray:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors

def matching(keypoints, descriptors, ratio=0.75):
    bf = cv2.BFMatcher()
    matchesList = {}
    pointsList = {}

    for i, j in itertools.combinations(range(len(descriptors)), 2):
        des1, des2 = descriptors[i], descriptors[j]
        kp1, kp2 = keypoints[i], keypoints[j]

        if des1 is None or des2 is None:
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < ratio * n.distance]

        print(f"[INFO] Found {len(good)} good matches {i} /// {j}")

        if len(good) < 8:
            continue  


        Point1 = np.float32([kp1[m.queryIdx].pt for m in good])
        Point2 = np.float32([kp2[m.trainIdx].pt for m in good])

        F, mask = cv2.findFundamentalMat(Point1, Point2, cv2.FM_RANSAC, 0.99, 3.0)

        if F is None or mask is None:
            continue

        mask = mask.ravel().astype(bool)
        GoodPoint1 = Point1[mask]
        GoodPoint2 = Point2[mask]

        if len(GoodPoint1) >= 8:
            matchesList[(i, j)] = good
            pointsList[(i, j)] = (GoodPoint1, GoodPoint2)

    return matchesList, pointsList

def visualize(img1, kp1, img2, kp2, matches, matchesLimit=50):
    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches[:matchesLimit], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save(pointsMatched, output):
    serialized = {
        f"{i}_{j}": points for (i, j), points in pointsMatched.items()
    }
    np.savez(output, **serialized)
    print(f"savet to {output}")

def main():
    directoryColor = "C:/Users/estus/Desktop/bakaproj/output/color"
    directoryGray = "C:/Users/estus/Desktop/bakaproj/output/gray"
    output_file = "output/matches/matches.npz"

    imgGray = gray(directoryGray)
    imgColored = gray(directoryColor)

    keypoints, descriptors = features(imgGray)

    print("Working")
    matches, pointsMatched = matching(keypoints, descriptors)

    print(f"Found {len(matches)} good pairs")

    for (i, j), inlier_matches in matches.items():
        visualize(imgColored[i], keypoints[i], imgColored[j], keypoints[j], inlier_matches)
        break

    save(pointsMatched, output_file)

if __name__ == "__main__":
    main()
