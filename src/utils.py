import json
import cv2
from PIL import ImageDraw


def load_annots(annots_file, annot_format):
    with open(annots_file, "r") as f:
        data = json.load(f)

    annots = {"text": [], "bbox": []}
    if annot_format == "STD":
        for text_block in data:
            annots["text"].append(text_block["text"])
            annots["bbox"].append(text_block["bbox"])

    elif annot_format == "ICPR22":
        for text_block in data["task2"]["output"]["text_blocks"]:
            annots["text"].append(text_block["text"])
            annots["bbox"].append(quad_to_box(text_block["polygon"]))

    elif annot_format == "EconBiz & CHIMIE-R":
        for text_block in data["textelements"]:
            annots["text"].append(text_block["content"])
            annots["bbox"].append(
                quad_to_box(
                    get_quad(text_block["boundingbox"], data["width"], data["height"])
                )
            )
    else:
        raise ValueError(f"Unknown annotation format: {annot_format}")

    return annots


def annotate_image(image, labeled_annots_data):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for b, label in zip(labeled_annots_data["bbox"], labeled_annots_data["labels"]):
        x0, y0, x1, y1 = b
        # Skip zero bboxes if needed
        if (x0, y0, x1, y1) == (0, 0, 0, 0):
            continue
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0 - 10), label, fill="red")

    return image


def normalize_bbox(bbox, size, type=None):
    if type == "box":
        height = int(bbox["height"])
        width = int(bbox["width"])
        left = max(0, bbox["x0"])
        top = max(0, bbox["y0"])
        right = left + width
        bottom = top + height
    if type == "polygon":
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]
    return [
        int(1000 * left / size[0]),
        int(1000 * top / size[1]),
        int(1000 * right / size[0]),
        int(1000 * bottom / size[1]),
    ]


def quad_to_box(quad):
    box = (max(0, quad["x0"]), max(0, quad["y0"]), quad["x2"], quad["y2"])
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


def get_quad(bbox, width, height):
    x0 = int(bbox["center_x"] - bbox["width"] / 2)
    x1 = int(bbox["center_x"] + bbox["width"] / 2)
    x2 = int(bbox["center_x"] + bbox["width"] / 2)
    x3 = int(bbox["center_x"] - bbox["width"] / 2)
    y0 = int(bbox["center_y"] - bbox["height"] / 2)
    y1 = int(bbox["center_y"] - bbox["height"] / 2)
    y2 = int(bbox["center_y"] + bbox["height"] / 2)
    y3 = int(bbox["center_y"] + bbox["height"] / 2)

    if bbox["orientation"] == 0:
        return {
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y0": y0,
            "y1": y1,
            "y2": y2,
            "y3": y3,
        }

    # rotate coordinates if orientation is not 0

    cx, cy = (int(width / 2), int(height / 2))

    bbox_tuple = [
        (x0, y0),
        (x1, y1),
        (x2, y2),
        (x3, y3),
    ]

    rotated_bbox = []

    for i, coord in enumerate(bbox_tuple):
        M = cv2.getRotationMatrix2D((cx, cy), bbox["orientation"], 1.0)
        v = [coord[0], coord[1], 1]
        adjusted_coord = np.matmul(M, v)
        rotated_bbox.insert(i, (adjusted_coord[0], adjusted_coord[1]))

    result = [int(x) for t in rotated_bbox for x in t]

    # make sure resulting bbox coordinates are within the range of the image
    for i, n in enumerate(result):
        if i % 2 == 0 and n > width:
            result[i] = width
        elif i % 2 == 1 and n > height:
            result[i] = height
        elif n < 0:
            result[i] = 0

    return {
        "x0": result[0],
        "x1": result[2],
        "x2": result[4],
        "x3": result[6],
        "y0": result[1],
        "y1": result[3],
        "y2": result[5],
        "y3": result[7],
    }
