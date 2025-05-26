import cv2
import os
import numpy as np

def processIM(folder, resize_width=1024):
    color = []
    gray = []
    names = []

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            
        
            height, width = img.shape[:2]
            scale = resize_width / float(width)
            resize = cv2.resize(img, (resize_width, int(height * scale)))
            grayi = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
            color.append(resize)
            gray.append(grayi)
            names.append(filename)
    return color, gray, names

if __name__ == "__main__":
    folder = "C:/Users/estus/Desktop/bakaproj/Images"
    colorIM, grayIM, names = processIM(folder)

    print(f"Loaded {len(colorIM)} images")


    os.makedirs("output/color", exist_ok=True)
    os.makedirs("output/gray", exist_ok=True)

    for i, name in enumerate(names):
        cv2.imwrite(os.path.join("output/color", name), colorIM[i])
        cv2.imwrite(os.path.join("output/gray", f"gray_{name}"), grayIM[i])
    print("Saved")
