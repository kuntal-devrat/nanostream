"""Real-World PASCAL VOC Dataset Loader and YOLO Exporter.

Downloads and parses PASCAL VOC real-world photos, extracting objects and
normalizing bounding boxes for NanoStream-OD and YOLOv8 training.
"""

import os
import pathlib
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Top 4 most relevant edge detection classes from PASCAL VOC
VOC_CLASSES = ("person", "car", "bicycle", "dog")
CLASS_TO_ID = {c: i for i, c in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES)

DATA_ROOT = pathlib.Path("data/pascal_voc")
YOLO_DIR = DATA_ROOT / "yolo_format"


def download_voc_if_needed():
    """Download PASCAL VOC 2007 dataset via torchvision if not present."""
    import torchvision
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    voc_raw = DATA_ROOT / "VOCdevkit" / "VOC2007"
    if not voc_raw.exists():
        print(f"[Dataset] Downloading PASCAL VOC 2007 to {DATA_ROOT}...")
        torchvision.datasets.VOCDetection(
            root=str(DATA_ROOT), year="2007", image_set="trainval", download=True
        )
        torchvision.datasets.VOCDetection(
            root=str(DATA_ROOT), year="2007", image_set="test", download=True
        )
        print("[Dataset] PASCAL VOC download complete!")
    return voc_raw


def parse_voc_annotation(xml_path: pathlib.Path):
    """Extract bounding boxes and class labels from a VOC XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_elem = root.find("size")
    width = float(size_elem.find("width").text)
    height = float(size_elem.find("height").text)

    boxes = []
    labels = []
    for obj in root.findall("object"):
        cname = obj.find("name").text.lower()
        if cname in CLASS_TO_ID:
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            # Convert to normalized cx, cy, w, h
            cx = ((xmin + xmax) / 2.0) / width
            cy = ((ymin + ymax) / 2.0) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            if w > 0.01 and h > 0.01:
                boxes.append([cx, cy, w, h])
                labels.append(CLASS_TO_ID[cname])

    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32), \
        np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64), \
        int(width), int(height)


class RealVOCDataset(Dataset):
    """High-throughput In-Memory Real VOC Dataset for NanoStream-OD."""

    def __init__(self, split="train", input_size=160, max_samples=None):
        self.input_size = input_size
        voc_dir = download_voc_if_needed()
        split_file = voc_dir / "ImageSets" / "Main" / f"{'trainval' if split == 'train' else 'test'}.txt"
        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        print(f"[Dataset] Loading {split} split ({len(ids)} candidate images)...")
        for img_id in ids:
            xml_p = voc_dir / "Annotations" / f"{img_id}.xml"
            img_p = voc_dir / "JPEGImages" / f"{img_id}.jpg"
            if xml_p.exists() and img_p.exists():
                boxes, labels, orig_w, orig_h = parse_voc_annotation(xml_p)
                if len(boxes) > 0:  # Only include images with target classes
                    self.samples.append((str(img_p), boxes, labels))
                    if max_samples and len(self.samples) >= max_samples:
                        break

        print(f"[Dataset] Loaded {len(self.samples)} valid {split} images containing: {VOC_CLASSES}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            gray = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

        # Convert to float tensor [-1, 1]
        x = (torch.from_numpy(gray).float().unsqueeze(0) / 127.5) - 1.0
        target = {
            "boxes_norm": torch.from_numpy(boxes.copy()),
            "labels": torch.from_numpy(labels.copy()),
        }
        return x, target


def export_voc_to_yolo():
    """Export VOC dataset to standardized YOLOv8 directory structure."""
    voc_dir = download_voc_if_needed()
    YOLO_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        img_dst = YOLO_DIR / "images" / split
        lbl_dst = YOLO_DIR / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        txt_name = "trainval.txt" if split == "train" else "test.txt"
        with open(voc_dir / "ImageSets" / "Main" / txt_name) as f:
            ids = [l.strip() for l in f if l.strip()]

        count = 0
        for img_id in ids:
            xml_p = voc_dir / "Annotations" / f"{img_id}.xml"
            img_p = voc_dir / "JPEGImages" / f"{img_id}.jpg"
            if xml_p.exists() and img_p.exists():
                boxes, labels, _, _ = parse_voc_annotation(xml_p)
                if len(boxes) > 0:
                    # Write image
                    img_out = img_dst / f"{img_id}.jpg"
                    if not img_out.exists():
                        im = cv2.imread(str(img_p))
                        cv2.imwrite(str(img_out), im)

                    # Write label file (class cx cy w h)
                    lbl_lines = [f"{lbl} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b, lbl in zip(boxes, labels)]
                    (lbl_dst / f"{img_id}.txt").write_text("\n".join(lbl_lines))
                    count += 1

        print(f"[YOLO Export] {split}: {count} images exported -> {img_dst}")

    # Write dataset.yaml
    yaml_content = f"""path: {YOLO_DIR.resolve()}
train: images/train
val: images/val

names:
"""
    for i, c in enumerate(VOC_CLASSES):
        yaml_content += f"  {i}: {c}\n"

    yaml_path = YOLO_DIR / "voc_edge.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[YOLO Export] dataset.yaml written -> {yaml_path}")
    return yaml_path
