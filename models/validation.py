import os
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from ClipCap import ClipCaptionModel
from typing import List
from utils import compose_discrete_prompts
from src.load_annotations import load_entities_text
from search import greedy_search, beam_search, opt_search
from retrieval_categories import clip_texts_embeddings, image_text_similarity, top_k_categories


def validation_uitviic(
    args,
    inpath: str,                             # path to pickle file with image features
    entities_text: List[str],                # entity names (categories)
    texts_embeddings: torch.Tensor,          # embeddings of entity names
    model: ClipCaptionModel,                 # trained captioning model
    tokenizer: AutoTokenizer,                # text tokenizer
) -> None:
    """
    Validation for UIT-ViIC dataset (COCO-like format).
    Uses pre-extracted image features (.pickle).
    """
    device = args.device
    print(f"🔹 Loading image features from: {inpath}")
    with open(inpath, 'rb') as infile:
        annotations = pickle.load(infile)  # [[image_name, image_features, [captions...]], ...]

    predicts = []
    for idx, item in tqdm(enumerate(annotations), total=len(annotations), desc="Generating captions"):
        image_id, image_features, captions = item
        image_features = image_features.float().unsqueeze(dim=0).to(device)

        continuous_embeddings = model.mapping_network(image_features).view(
            -1, args.continuous_prompt_length, model.gpt_hidden_size
        )

        if args.using_hard_prompt:
            print(image_features.shape)
            print(texts_embeddings.shape)
            logits = image_text_similarity(
                texts_embeddings=texts_embeddings,
                images_features=image_features,
                temperature=args.temperature
            )
            
            detected_objects, _ = top_k_categories(
                entities_text, logits, args.top_k, args.threshold
            )
            detected_objects = detected_objects[0]
            
            print(f"🧠 [{idx}] {image_id} → Detected: {detected_objects}")

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

        # ===== Text generation =====
        if 'gpt' in args.language_model:
            if not args.using_greedy_search:
                sentence = beam_search(
                    embeddings=embeddings,
                    tokenizer=tokenizer,
                    beam_width=args.beam_width,
                    model=model.gpt
                )[0]
            else:
                sentence = greedy_search(embeddings, tokenizer, model.gpt)
        else:
            sentence = opt_search(
                prompts=args.text_prompt,
                embeddings=embeddings,
                tokenizer=tokenizer,
                beam_width=args.beam_width,
                model=model.gpt
            )[0]

        predicts.append({
            "split": "valid",
            "image_name": image_id,
            "captions": captions,
            "prediction": sentence
        })

    # ===== Save results =====
    os.makedirs(args.out_path, exist_ok=True)
    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
    with open(out_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(predicts, outfile, indent=4, ensure_ascii=False)
    print(f"Saved predictions to {out_json_path}")


@torch.no_grad()
def main(args) -> None:
    device = args.device
    clip_name = args.clip_model.replace('/', '')
    clip_hidden_size = 1024  # ViT-B/32 -> 512-dim embeddings

    # ===== Load entity vocab and embeddings =====
    print(f"🔹 Loading entities from {args.entities_file}")
    entities_text = load_entities_text(
        args.name_of_entities_text,
        args.entities_file,
        not args.disable_all_entities
    )

    emb_dir = os.path.dirname(args.entities_file)
    if args.prompt_ensemble:
        emb_path = os.path.join(emb_dir, f"{args.name_of_entities_text}_{clip_name}_with_ensemble.pickle")
    else:
        emb_path = os.path.join(emb_dir, f"{args.name_of_entities_text}_{clip_name}.pickle")

    print(f"🔹 Generating text embeddings with {args.clip_model}")
    texts_embeddings = clip_texts_embeddings(entities_text, emb_path, device=device)

    texts_embeddings = texts_embeddings.float()
    texts_embeddings = texts_embeddings / texts_embeddings.norm(dim=-1, keepdim=True)
    print(f"Text embeddings normalized and saved at: {emb_path}")

    # ===== Load captioning model =====
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(
        args.continuous_prompt_length,
        args.clip_project_length,
        clip_hidden_size,
        gpt_type=args.language_model
    )
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model.to(device)
    model.eval()

    # ===== Determine input file =====
    if args.using_image_features:
        if args.image_features_path and os.path.exists(args.image_features_path):
            inpath = args.image_features_path
        else:
            inpath = args.path_of_val_datasets[:-5] + f'_{clip_name}_with_features.pickle'
    else:
        raise ValueError("UIT-ViIC validation requires --using_image_features for speed.")

    # ===== Run validation =====
    validation_uitviic(args, inpath, entities_text, texts_embeddings, model, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clip_model', default="BAAI/AltCLIP-m18")
    parser.add_argument('--language_model', default='NlpHUST/gpt2-vietnamese')
    parser.add_argument('--continuous_prompt_length', type=int, default=10)
    parser.add_argument('--clip_project_length', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--using_image_features', action='store_true', default=True)
    parser.add_argument('--image_features_path', type=str, default=None,
                        help='Path to pre-extracted image features (.pickle)')
    parser.add_argument('--name_of_datasets', default='uitviic')
    parser.add_argument('--path_of_val_datasets', default='./dataset/UIT-ViIC/uitviic_captions_val2017.json')
    parser.add_argument('--entities_file', default='./dataset/vietnamese_categories.json')
    parser.add_argument('--disable_all_entities', action='store_true', default=False)
    parser.add_argument('--name_of_entities_text', default='vietnamese_entities')
    parser.add_argument('--prompt_ensemble', action='store_true', default=True)
    parser.add_argument('--weight_path', default='./checkpoints/viecap_vi_multilingual/vietnamese-0019.pt')
    parser.add_argument('--out_path', default='./checkpoints/viecap_vi_multilingual/')
    parser.add_argument('--using_hard_prompt', action='store_true', default=True)
    parser.add_argument('--soft_prompt_first', action='store_true', default=True)
    parser.add_argument('--only_hard_prompt', action='store_true', default=False)
    parser.add_argument('--using_greedy_search', action='store_true', default=False)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--text_prompt', type=str, default=None)
    args = parser.parse_args()

    print('args: {}\n'.format(vars(args)))
    main(args)
