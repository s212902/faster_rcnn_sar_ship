import pandas as pd
import os
import re
from PIL import Image

DATASET_PATH = "G:/My Drive/DTU/Semester 1/Deep Learning/Project/ship_dataset_v0/train"

def getFiles(path):
    return [f for f in os.listdir(path) if re.match("^.+\.txt$", f)]

def getSizes(files):
    widths = []
    heights = []
    for i in range(len(files)):
        if i % 1000 == 0:
            print(i)
        filePath = DATASET_PATH + "/" + files[i].split(".")[0] + ".jpg"
        im = Image.open(filePath)
        width, height = im.size
        if width != 256 or height != 256:
            print(filePath)
        widths.append(width)
        heights.append(height)

    return widths, heights
        
def getBoundingBoxPixelFromFile(filename, width, height, i):
    if i % 1000 == 0:
        print(i)
    with open(DATASET_PATH + "/" + filename) as f:
        boxes = []
        lines = f.readlines()

        for i in range(len(lines)):
            lineData = lines[i].strip().split(" ")
            [center_x, center_y, w, h] = lineData[1:]
            center_x = float(center_x) * width
            center_y = float(center_y) * height
            w = float(w) * width
            h = float(h) * height
            x = center_x - w / 2.0
            y = center_y - h / 2.0
            boxes.append([x, y, w, h])
        return boxes

if __name__ == "__main__":
    files = getFiles(DATASET_PATH)
    source = ['ssdd'] * len(files)
    width_compressed, height_compressed = getSizes(files)
    bbox_compressed = [getBoundingBoxPixelFromFile(files[i], width_compressed[i], height_compressed[i], i) for i in range(len(files))]

    image_ids = []
    bbox = []
    width = []
    height = []
    for i in range(len(bbox_compressed)):
        for j in range(len(bbox_compressed[i])):
            image_ids.append(files[i].split('.')[0])
            bbox.append(bbox_compressed[i][j])
            width.append(width_compressed[i])
            height.append(height_compressed[i])
    source = ['ssdd'] * len(bbox)

    data = {'image_id': image_ids, 'width': width, 'height': height, 'bbox': bbox, 'source': source}
    data_df = pd.DataFrame(data)
    data_df.to_csv(DATASET_PATH + "/train.csv", index=False)

