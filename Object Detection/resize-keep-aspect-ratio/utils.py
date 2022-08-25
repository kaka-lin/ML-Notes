import cv2
import xml.etree.ElementTree as ET


def parse_annotation(file):
    tree = ET.parse(file)
    root = tree.getroot()

    bbox = [0] * 4
    for boxes in root.iter('object'):
        for box in boxes.findall('bndbox'):
            bbox[0] = float(box.find('xmin').text)
            bbox[1] = float(box.find('ymin').text)
            bbox[2] = float(box.find('xmax').text)
            bbox[3] = float(box.find('ymax').text)

    return bbox
