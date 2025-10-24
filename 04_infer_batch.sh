export PYTHONPATH=$(pwd)

python infer_batch.py \
--image_path ./dataset/UIT-ViIC/images/val \
--weight_path ./checkpoints/viecap_vi_multilingual_v4/vietnamese-0034.pt \
--soft_prompt_first \
--beam_width 5
