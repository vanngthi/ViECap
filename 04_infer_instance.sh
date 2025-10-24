export PYTHONPATH=$(pwd)

python models/infer_instance.py \
  --image_path ./dataset/UIT-ViIC/images/000000205086.jpg \
  --weight_path ./checkpoints/viecap_vi_multilingual_v4/vietnamese-004.pt \
  --soft_prompt_first \
  --beam_width 5
