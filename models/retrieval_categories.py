import os
import torch
import pickle
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers import AutoProcessor, AutoModel


@torch.inference_mode()
def clip_texts_embeddings(
    texts: List[str],
    outpath: str = '',
    device: Optional[str] = None,
    batch_size: int = 32,
    model_id: str = "BAAI/AltCLIP-m18"
) -> torch.Tensor:
    """
    Generate multilingual CLIP text embeddings using BAAI/AltCLIP-m18.
    Args:
        texts: list of entity names (e.g. ['người đàn ông', 'xe hơi', ...])
        outpath: where to save or load cached embeddings
        device: CUDA or CPU
        batch_size: embedding batch size
        model_id: AltCLIP model name
    Return:
        Tensor (num_entities, 768) normalized embeddings
    """
    if outpath and os.path.exists(outpath):
        print(f"📦 Loading cached text embeddings from {outpath}")
        with open(outpath, 'rb') as infile:
            return pickle.load(infile)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating text embeddings with {model_id} on {device}")

    # Load model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # Add a simple Vietnamese prompt to make text more CLIP-like
    prompt_texts = [f"Một bức ảnh về {t}" for t in texts]

    all_embeddings = []
    for i in range(0, len(prompt_texts), batch_size):
        batch = prompt_texts[i:i + batch_size]

        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        outputs = model.get_text_features(**inputs)
        outputs = F.normalize(outputs, dim=-1)

        all_embeddings.append(outputs.cpu())

    text_embeddings = torch.cat(all_embeddings, dim=0)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    if outpath:
        dirpath = os.path.dirname(outpath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(outpath, 'wb') as outfile:
            pickle.dump(text_embeddings, outfile)
        print(f"Saved text embeddings → {outpath}")

    return text_embeddings

@torch.no_grad()
def image_text_similarity(
    *,
    texts_embeddings: torch.Tensor,
    images_features: torch.Tensor,
    temperature: float = 0.01,
) -> torch.Tensor:
    """
    Compute similarity between image and text embeddings (CLIP multilingual).
    Args:
        texts_embeddings: (num_categories, dim)
        images_features: (num_images, dim)
    Return:
        probs (num_images, num_categories)
    """
    assert images_features is not None, "Need precomputed image_features."

    device = images_features.device
    texts_embeddings = texts_embeddings.to(device).float()
    images_features = images_features.to(device).float()

    # Normalize if not already normalized
    if not torch.allclose(images_features.norm(dim=-1), torch.ones_like(images_features.norm(dim=-1)), atol=1e-2):
        images_features = F.normalize(images_features, dim=-1)
    if not torch.allclose(texts_embeddings.norm(dim=-1), torch.ones_like(texts_embeddings.norm(dim=-1)), atol=1e-2):
        texts_embeddings = F.normalize(texts_embeddings, dim=-1)

    logits = torch.matmul(images_features, texts_embeddings.T) / temperature
    return torch.softmax(logits, dim=-1)

# @torch.no_grad()
# def image_text_similarity(
#     texts_embeddings: torch.Tensor,
#     images_features: torch.Tensor,
#     temperature: float = 0.01,
# ) -> torch.Tensor:
#     """
#     Compute similarity between precomputed image and text embeddings.
#     Args:
#         texts_embeddings: (num_categories, dim)
#         images_features: (num_images, dim)
#     Return:
#         probs (num_images, num_categories)
#     """
#     assert images_features is not None, "Need precomputed image_features."

#     images_features = F.normalize(images_features.float(), dim=-1)
#     texts_embeddings = F.normalize(texts_embeddings.float(), dim=-1)

#     device = images_features.device
#     images_features = images_features.float().to(device)
#     texts_embeddings = texts_embeddings.float().to(device)

#     logits = torch.matmul(images_features, texts_embeddings.T) / temperature
#     return torch.softmax(logits, dim=-1)


def top_k_categories(
    texts: List[str],
    logits: torch.Tensor,
    top_k: int = 5,
    threshold: float = 0.0
) -> Tuple[List[List[str]], torch.Tensor]:
    """
    Select top-k categories for each image.
    """
    top_k_probs, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
    top_k_texts = []
    for i in range(len(top_k_probs)):
        per_image_texts = [
            texts[top_k_indices[i][j]]
            for j in range(top_k)
            if top_k_probs[i][j] >= threshold
        ]
        top_k_texts.append(per_image_texts)
    return top_k_texts, top_k_probs