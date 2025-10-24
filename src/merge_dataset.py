import pickle
import torch

with open('./dataset/Flick_sportball/flick_captions_train_with_entities.pickle', 'rb') as f:
    flick = pickle.load(f)
with open('./dataset/UIT-ViIC/uitviic_captions_train_with_features.pickle', 'rb') as f:
    uit = pickle.load(f)

# lấy dimension của feature từ UIT-ViIC (vd: 768 hoặc 1024)
clip_dim = uit[0][2].shape[-1] if len(uit[0]) == 3 else 768

# nếu flick không có clip feature, thêm tensor zero
merged = []
for item in flick:
    if len(item) == 2:
        entities, caption = item
        clip_feat = torch.zeros(clip_dim)   # placeholder
        merged.append([entities, caption, clip_feat])
    else:
        merged.append(item)
for item in uit:
    merged.append(item)

print(f" Tổng mẫu: {len(merged)} ({len(flick)} Flickr + {len(uit)} UIT-ViIC)")
with open('./dataset/viecap_train_with_features.pickle', 'wb') as f:
    pickle.dump(merged, f)
