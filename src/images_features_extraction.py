import os
import json
import pickle
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch.nn.functional as F

@torch.no_grad()
def extract_features(model, processor, annotations, rootpath, outpath, device):
    """
    Extract image features using BAAI/AltCLIP-m18.
    Each entry: [image_name, image_feature (768-d), captions]
    """
    results = []
    for image_name, captions in tqdm(annotations.items(), desc=f"Extracting {len(annotations)} images"):
        image_path = os.path.join(rootpath, image_name)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to(device)

            image_features = model.get_image_features(**image_inputs)
            image_features = F.normalize(image_features, dim=-1)[0].cpu()  # 768-dim

            results.append([image_name, image_features, captions])
        except Exception as e:
            print(f"[ERROR] Failed to process {image_path}: {e}")

    # Save features
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} image features to {outpath}")


def coco_to_dict(raw_json):
    id_to_filename = {img["id"]: img["file_name"] for img in raw_json["images"]}
    result = defaultdict(list)
    for ann in raw_json["annotations"]:
        fname = id_to_filename.get(ann["image_id"])
        if fname is not None:
            result[fname].append(ann["caption"])
    return dict(result)


if __name__ == "__main__":
    model_name = "BAAI/AltCLIP-m18"
    dataset_name = "UIT-ViIC"
    base_dir = f"./dataset/{dataset_name}"
    image_root = os.path.join(base_dir, "images")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    json_files = {
        # "train": os.path.join(base_dir, "uitviic_captions_train2017.json"),
        "val": os.path.join(base_dir, "uitviic_captions_val2017.json"),
        # "test": os.path.join(base_dir, "uitviic_captions_test2017.json"),
    }

    print(f"🔹 Loading model: {model_name} on {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    for split, json_path in json_files.items():
        outpath = os.path.join(base_dir, f"uitviic_images_{split}_with_features.pickle")
        if os.path.exists(outpath):
            print(f"Found existing {outpath}")
            try:
                with open(outpath, "rb") as f:
                    data = pickle.load(f)
                print(f"Loaded {len(data)} samples from existing file.")
                print("Showing first 3 samples:\n")
                for i, sample in enumerate(data[:3]):
                    img_name, feat, caps = sample
                    print(f"  [{i}] {img_name}")
                    print(f"      features: {tuple(feat.shape)}, dtype={feat.dtype}")
                    print(f"      captions: {caps[:2] if len(caps)>2 else caps}\n")
            except Exception as e:
                print(f"[ERROR] Cannot read existing file {outpath}: {e}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        annotations = coco_to_dict(raw)
        print(f"{split}: {len(annotations)} images")

        extract_features(model, processor, annotations, image_root, outpath, device)
