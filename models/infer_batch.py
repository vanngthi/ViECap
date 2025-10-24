import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, AutoModel

from ClipCap import ClipCaptionModel
from utils import compose_discrete_prompts
from src.load_annotations import load_entities_text
from search import greedy_search, beam_search
from retrieval_categories import clip_texts_embeddings, image_text_similarity, top_k_categories


@torch.no_grad()
def main(args):
    device = args.device
    clip_hidden_size = 1024  # AltCLIP-m18

    # ===== Load entities =====
    print(f"🔹 Loading entities from {args.entities_file}")
    entities_text = load_entities_text(
        args.name_of_entities_text, args.entities_file, not args.disable_all_entities
    )

    # ===== Precompute text embeddings =====
    clip_name = args.clip_model.replace("/", "_")
    emb_path = os.path.join(
        os.path.dirname(args.entities_file),
        f"{args.name_of_entities_text}_{clip_name}.pickle"
    )
    print(f"🔹 Loading / caching entity embeddings → {emb_path}")
    texts_embeddings = clip_texts_embeddings(entities_text, emb_path, device=device)

    # ===== Load captioning model =====
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(
        args.continuous_prompt_length,
        args.clip_project_length,
        clip_hidden_size,
        gpt_type=args.language_model,
    )
    model.load_state_dict(torch.load(args.weight_path, map_location=device), strict=False)
    model.to(device).eval()

    # ===== Load AltCLIP encoder =====
    processor = AutoProcessor.from_pretrained(args.clip_model)
    clip_model = AutoModel.from_pretrained(args.clip_model).to(device).eval()

    # ===== Collect all images in folder =====
    images_list = sorted([
        os.path.join(args.image_path, img)
        for img in os.listdir(args.image_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    print(f"Found {len(images_list)} images in {args.image_path}")

    # ===== Inference loop =====
    predicts = []
    for im_path in tqdm(images_list, desc="🖼️ Generating captions"):
        try:
            # Encode image
            image = Image.open(im_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = F.normalize(image_features.float(), dim=-1)

            # Map to continuous prefix
            continuous_embeddings = model.mapping_network(image_features).view(
                -1, args.continuous_prompt_length, model.gpt_hidden_size
            )

            # ===== Hard prompt (entity-aware) =====
            if args.using_hard_prompt:
                logits = image_text_similarity(
                    texts_embeddings,
                    temperature=args.temperature,
                    images_features=image_features,
                )
                detected_objects, _ = top_k_categories(
                    entities_text, logits, args.top_k, args.threshold
                )
                detected_objects = detected_objects[0]

                discrete_tokens = compose_discrete_prompts(
                    tokenizer, detected_objects
                ).unsqueeze(0).to(device)
                discrete_embeddings = model.word_embed(discrete_tokens)

                if args.only_hard_prompt:
                    embeddings = discrete_embeddings
                elif args.soft_prompt_first:
                    embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
                else:
                    embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)
            else:
                embeddings = continuous_embeddings

            # ===== Caption generation =====
            if not args.using_greedy_search:
                sentence = beam_search(
                    embeddings=embeddings,
                    tokenizer=tokenizer,
                    beam_width=args.beam_width,
                    model=model.gpt,
                )[0]
            else:
                sentence = greedy_search(embeddings, tokenizer, model.gpt)

            predicts.append({
                "image_name": os.path.basename(im_path),
                "prediction": sentence
            })

        except Exception as e:
            print(f"[WARN] Skipping {im_path}: {e}")
            continue

    # ===== Save all results =====
    out_path = os.path.join(args.image_path, "predictions_viecap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predicts, f, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(predicts)} captions → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_model", default="BAAI/AltCLIP-m18")
    parser.add_argument("--language_model", default="NlpHUST/gpt2-vietnamese")
    parser.add_argument("--continuous_prompt_length", type=int, default=10)
    parser.add_argument("--clip_project_length", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--disable_all_entities", action="store_true", default=False)
    parser.add_argument("--name_of_entities_text", default="vietnamese_entities")
    parser.add_argument("--weight_path", default="./checkpoints/viecap_vi_multilingual_v4/vietnamese-0034.pt")
    parser.add_argument("--entities_file", default="./dataset/vietnamese_categories.json")
    parser.add_argument("--image_path", default="./demo_images")
    parser.add_argument("--using_hard_prompt", action="store_true", default=True)
    parser.add_argument("--soft_prompt_first", action="store_true", default=True)
    parser.add_argument("--only_hard_prompt", action="store_true", default=False)
    parser.add_argument("--using_greedy_search", action="store_true", default=False)
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    print("args:", vars(args))
    main(args)
