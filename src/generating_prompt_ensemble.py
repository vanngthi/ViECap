import os
import torch
import pickle
from tqdm import tqdm
from typing import List
from sentence_transformers import SentenceTransformer
from load_annotations import load_entities_text

@torch.no_grad()
def generate_ensemble_prompt_embeddings(
    device: str,
    model_name: str,
    entities: List[str],
    prompt_templates: List[str],
    outpath: str,
):
    # Nếu file đã tồn tại thì load lại luôn
    if os.path.exists(outpath):
        print(f"Found existing embeddings at {outpath}")
        with open(outpath, 'rb') as infile:
            return pickle.load(infile)

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    embeddings = []
    for entity in tqdm(entities, desc="Encoding entities"):
        # Ví dụ: ["một bức ảnh về con mèo.", "ảnh chụp con mèo."]
        texts = [template.format(entity) for template in prompt_templates]
        text_embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device
        )
        # Lấy trung bình giữa các template
        mean_emb = text_embeddings.mean(dim=0)
        mean_emb /= mean_emb.norm()
        embeddings.append(mean_emb.cpu())

    embeddings = torch.stack(embeddings, dim=0)
    with open(outpath, 'wb') as outfile:
        pickle.dump(embeddings, outfile)

    print(f"Saved embeddings to {outpath}")
    return embeddings

if __name__ == '__main__':
    # prompt templates – bạn có thể dùng tiếng Việt để hợp hơn
    prompt_templates = [
        'một bức ảnh về {}.',
        'ảnh chụp {}.',
        'hình ảnh của {}.',
        'một bức tranh về {}.',
        '{} trong đời thực.',
        'ảnh {} trong trò chơi.',
        '{} trong môi trường tự nhiên.'
    ]

    # Load entity tiếng Việt
    entities = load_entities_text(
        'vietnamese_entities',
        '../dataset/vietnamese_categories.json'
    )

    device = 'cuda:0'
    model_name = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
    outpath = '../dataset/uitviic_embeddings_with_ensemble.pickle'

    embeddings = generate_ensemble_prompt_embeddings(device, model_name, entities, prompt_templates, outpath)

    print("Số lượng entity:", len(entities))
    print("Embedding shape:", embeddings.shape)
