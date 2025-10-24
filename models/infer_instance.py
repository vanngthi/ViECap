import os
import json
import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel
import torch.nn.functional as F
from ClipCap import ClipCaptionModel
from utils import compose_discrete_prompts
from search import greedy_search, beam_search
from src.load_annotations import load_entities_text


@torch.no_grad()
def extract_entities_from_image(image_path, processor, clip_model, vocab_list, device, top_k=5):
    """
    Tự động chọn các entity phù hợp nhất cho ảnh bằng CLIP similarity.
    """
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    image_feat = clip_model.get_image_features(**image_inputs)
    image_feat = F.normalize(image_feat, dim=-1)

    # text embeddings cho vocab
    text_inputs = processor(text=vocab_list, return_tensors="pt", padding=True).to(device)
    text_feat = clip_model.get_text_features(**text_inputs)
    text_feat = F.normalize(text_feat, dim=-1)

    # cosine similarity
    sims = (image_feat @ text_feat.T).squeeze(0)
    topk_indices = sims.topk(top_k).indices.tolist()
    entities = [vocab_list[i] for i in topk_indices]
    return entities


@torch.no_grad()
def main(args):
    device = args.device
    clip_hidden_size = 1024  # AltCLIP-m18 → 768 dim
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    # ===== Load caption model =====
    model = ClipCaptionModel(
        args.continuous_prompt_length,
        args.clip_project_length,
        clip_hidden_size,
        gpt_type=args.language_model,
    )
    model.load_state_dict(torch.load(args.weight_path, map_location=device), strict=False)
    model.to(device).eval()

    # ===== Load CLIP encoder (AltCLIP) =====
    processor = AutoProcessor.from_pretrained(args.clip_model)
    clip_model = AutoModel.from_pretrained(args.clip_model).to(device).eval()

    # ===== Load vocab (for hard prompt) =====
    vocab_path = args.path_of_objects_vocabs
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = json.load(f)
    print(f"Loaded {len(vocab_list)} vocab entities")

    # ===== Encode image features =====
    image_path = args.image_path
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
    image_features = F.normalize(image_features, dim=-1)
    continuous_embeddings = model.mapping_network(image_features).view(
        -1, args.continuous_prompt_length, model.gpt_hidden_size
    )

    # ===== Auto-generate hard prompt =====
    detected_objects = extract_entities_from_image(
        image_path, processor, clip_model, vocab_list, device, top_k=args.top_k_entities
    )
    print(f"Auto-detected entities: {detected_objects}")

    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(0).to(device)
    discrete_embeddings = model.word_embed(discrete_tokens)

    if args.only_hard_prompt:
        embeddings = discrete_embeddings
    elif args.soft_prompt_first:
        embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
    else:
        embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)

    # ===== Generate caption =====
    sentence = beam_search(
        embeddings=embeddings,
        tokenizer=tokenizer,
        beam_width=args.beam_width,
        model=model.gpt,
    )[0]

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Caption: {sentence}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_model", default="BAAI/AltCLIP-m18")
    parser.add_argument("--language_model", default="NlpHUST/gpt2-vietnamese")
    parser.add_argument("--continuous_prompt_length", type=int, default=10)
    parser.add_argument("--clip_project_length", type=int, default=10)
    parser.add_argument("--weight_path", default="./checkpoints/viecap_vi_multilingual_v4/vietnamese-0034.pt")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--path_of_objects_vocabs", default="./dataset/vietnamese_categories.json")
    parser.add_argument("--soft_prompt_first", action="store_true", default=True)
    parser.add_argument("--only_hard_prompt", action="store_true", default=False)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--top_k_entities", type=int, default=5)
    args = parser.parse_args()

    print("args:", vars(args))
    main(args)
