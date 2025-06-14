from utils import load_annots, normalize_bbox, annotate_image
from transformers import AutoModelForTokenClassification, AutoProcessor
import os
import pandas as pd


model = AutoModelForTokenClassification.from_pretrained(
    "nassimb0u/chart-text-role-classification-model"
)
processor = AutoProcessor.from_pretrained(
    "nassimb0u/chart-text-role-classification-model"
)


def process_image_and_annot(image, annots_data):
    prepro_annots_data = {"text": [], "bbox": []}

    for b, t in zip(annots_data["bbox"], annots_data["text"]):
        prepro_annots_data["bbox"].append(normalize_bbox(b, image.size, type="polygon"))
        prepro_annots_data["text"].append(t)
    return image.convert("RGB"), annots_data


def perform_inference(image, annots_file, annots_format):
    annots_data = load_annots(annots_file, annots_format)
    _, prepro_annots_data = process_image_and_annot(
        image,
        annots_data,
    )

    encoding = processor(
        image,
        prepro_annots_data["text"],
        boxes=prepro_annots_data["bbox"],
        return_tensors="pt",
    )
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1)

    labels = [model.config.id2label[idx.item()] for idx in predictions[0]]
    mask = []
    for i in range(encoding["bbox"].shape[1]):
        zero = True
        equal_to_pred = True
        for j in range(encoding["bbox"].shape[2]):
            if encoding["bbox"][0][i][j] != 0:
                zero = False
            if i > 0 and encoding["bbox"][0][i - 1][j] != encoding["bbox"][0][i][j]:
                equal_to_pred = False

        mask.append(not (zero or equal_to_pred))

    annots_data["labels"] = [label for (m, label) in zip(mask, labels) if m]

    image = annotate_image(image, annots_data)

    out_file_name = f"out/{os.path.basename(annots_file).split(".")[0]}_labeled.json"
    df = pd.DataFrame(annots_data)
    df.to_json(out_file_name, orient="records", lines=False, indent=2)

    return image, out_file_name
