import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET


# Load XML
tree = ET.parse("data/input/annotations.xml")
root = tree.getroot()

# Output folder
out_dir = "data/input/test1/ground_truth"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
os.makedirs(out_dir, exist_ok=True)

for image_tag in root.findall("image"):
    name = image_tag.attrib["name"]
    width = int(image_tag.attrib["width"])
    height = int(image_tag.attrib["height"])

    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in image_tag.findall("polygon"):
        points_str = poly.attrib["points"]
        pts = np.array([
            [float(x), float(y)] for x, y in
            (p.split(",") for p in points_str.split(";"))
        ], np.int32)

        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255) 

    base_name = os.path.splitext(name)[0] + ".png"
    out_path = os.path.join(out_dir, base_name)
    cv2.imwrite(out_path, mask)