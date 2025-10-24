# import json
# import torch
# from PIL import Image
# from sentence_transformers import SentenceTransformer, util
# from collections import defaultdict
# import random
# import os
# import torch
# import torch.nn.functional as F

# # ====== CONFIG ======
# json_path = "./dataset/UIT-ViIC/uitviic_captions_val2017.json"
# root_image_dir = "./dataset/UIT-ViIC/images/"  # thư mục chứa ảnh
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # ====== LOAD MODEL ======
# model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", device=device)
# # text_embs = model.encode(["đàn_ông", "đàn ông"], convert_to_tensor=True, device=device)
# # print(text_embs)
# # ====== LOAD DATA ======
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Map ảnh → list captions
# image_to_captions = defaultdict(list)
# for ann in data["annotations"]:
#     image_to_captions[ann["image_id"]].append(ann["caption"])

# # Map id → file_name
# id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

# # ====== CHỌN NGẪU NHIÊN 3 ẢNH ======
# sample_ids = random.sample(list(image_to_captions.keys()), 5)

# # for image_id in sample_ids:
# #     image_path = os.path.join(root_image_dir, id_to_file[image_id])
# #     captions = image_to_captions[image_id]

# #     # Load và encode ảnh + caption
# #     image_emb = model.encode([image_path], convert_to_tensor=True, device=device)

# #     text_embs = model.encode(captions, convert_to_tensor=True, device=device)
# #     print(captions)
# #     # # Cosine similarity
# #     # sims = util.cos_sim(image_emb, text_embs)[0]
# #     # print(f"\n🖼️ {id_to_file[image_id]}")
# #     # for c, s in zip(captions, sims):
# #     #     print(f"  {s.item():.4f}  →  {c}")

# id = 410933
# print(id_to_file[id])
# image_path = os.path.join(root_image_dir, id_to_file[id])
# captions = image_to_captions[id]

# image_emb = model.encode([image_path], convert_to_tensor=True, device=device)

# text_embs = model.encode(captions, convert_to_tensor=True, device=device)

# sims = util.cos_sim(image_emb, text_embs)[0]
# for c, s in zip(captions, sims):
#     print(f"  {s.item():.4f}  →  {c}")

# entities = ["cậu bé", "người", "cô gái", "điền kinh", "cầu thủ", "bóng", "vợt tennis", "bức ảnh", "chậu cảnh"]
# text_embs = model.encode(entities, convert_to_tensor=True, device=device)

# center = text_embs.mean(dim=0, keepdim=True)

# # Center-debias
# text_embs_c = F.normalize(text_embs - center, dim=-1)
# image_emb_c = F.normalize(image_emb - center, dim=-1)

# # Cosine
# sims = util.cos_sim(image_emb_c, text_embs_c)[0]

# # Z-score rescale
# sims_z = (sims - sims.mean()) / (sims.std() + 1e-6)
# probs = torch.softmax(sims_z, dim=-1)

# for c, s, p in zip(entities, sims_z, probs):
#     print(f"{p.item():.3f}  {s.item():+.3f}  →  {c}")
    
# # sims = util.cos_sim(image_emb, text_embs)[0]
# # sims_z = (sims - sims.mean()) / (sims.std() + 1e-6)

# # probs = torch.softmax(sims_z, dim=-1)

# # for c, s, p in zip(entities, sims_z, probs):
# #     print(f"{p.item():.3f}  {s.item():+.3f}  →  {c}")
    
# # for c, s in zip(entities, sims):
# #     print(f"  {s.item():.4f}  →  {c}")


import json
import torch
from PIL import Image
from collections import defaultdict
import random
import os
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

# ====== CONFIG ======
json_path = "./dataset/UIT-ViIC/uitviic_captions_val2017.json"
root_image_dir = "./dataset/UIT-ViIC/images/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== LOAD MODEL ======
model_id = "BAAI/AltCLIP-m18"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

# ====== LOAD DATA ======
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Map ảnh → list captions
image_to_captions = defaultdict(list)
for ann in data["annotations"]:
    image_to_captions[ann["image_id"]].append(ann["caption"])

# Map id → file_name
id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

# ====== CHỌN NGẪU NHIÊN 5 ẢNH ======
sample_ids = random.sample(list(image_to_captions.keys()), 5)

for image_id in sample_ids:
    image_path = os.path.join(root_image_dir, id_to_file[image_id])
    captions = image_to_captions[image_id]
    captions = captions + ["cậu bé", "cô gái", "vợt", "bàn", "mây", "chậu hoa", "đàn ông", "người phụ nữ", "cầu thủ"]

    # === Encode image ===
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**image_inputs)
        image_emb = F.normalize(image_emb, dim=-1)

    # === Encode text ===
    text_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embs = model.get_text_features(**text_inputs)
        text_embs = F.normalize(text_embs, dim=-1)

    # === Cosine similarity ===
    sims = (image_emb @ text_embs.T)[0]

    print(f"\n🖼️ {id_to_file[image_id]}")
    for c, s in zip(captions, sims):
        print(f"  {s.item():.4f}  →  {c}")



# import json
# import torch
# from PIL import Image
# from collections import defaultdict
# import random
# import os
# import torch.nn.functional as F
# from multilingual_clip import pt_multilingual_clip
# from transformers import AutoTokenizer
# import clip

# # ====== CONFIG ======
# json_path = "./dataset/UIT-ViIC/uitviic_captions_val2017.json"
# root_image_dir = "./dataset/UIT-ViIC/images/"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # ====== LOAD M-CLIP ======
# model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
# model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model.eval()

# clip_model, preprocess = clip.load("ViT-B/32", device=device)
# # ====== LOAD DATA ======
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Map ảnh → list captions
# image_to_captions = defaultdict(list)
# for ann in data["annotations"]:
#     image_to_captions[ann["image_id"]].append(ann["caption"])

# # Map id → file_name
# id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

# # ====== CHỌN NGẪU NHIÊN 5 ẢNH ======
# sample_ids = random.sample(list(image_to_captions.keys()), 5)

# for image_id in sample_ids:
#     image_path = os.path.join(root_image_dir, id_to_file[image_id])
#     captions = image_to_captions[image_id]
#     captions += ["cậu bé", "cô gái", "vợt", "bàn", "mây", "chậu hoa", "đàn ông", "người phụ nữ", "cầu thủ"]

#     # === Encode image ===
#     image = Image.open(image_path).convert("RGB")
#     image = preprocess(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         image_emb = clip_model.encode_image(image)
#         image_emb = F.normalize(image_emb, dim=-1)

#     # === Encode text ===
#     with torch.no_grad():
#         text_embs = model(captions, tokenizer)
#         text_embs = F.normalize(text_embs, dim=-1).to(device)
#         # text_embs = F.normalize(text_embs, dim=-1)

#     # === Cosine similarity ===
#     image_emb = image_emb.float()
#     text_embs = text_embs.float()
#     sims = (image_emb @ text_embs.T)[0]

#     print(f"\n🖼️ {id_to_file[image_id]}")
#     for c, s in zip(captions, sims):
#         print(f"  {s.item():.4f}  →  {c}")
