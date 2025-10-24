export PYTHONPATH=$(pwd)

python models/validation.py \
        --device cuda:0 \
        --clip_model "BAAI/AltCLIP-m18" \
        --language_model NlpHUST/gpt2-vietnamese \
        --using_image_features \
        --name_of_datasets uit_vilc \
        --image_features_path ./dataset/UIT-ViIC/uitviic_images_val_with_features.pickle \
        --entities_file ./dataset/vietnamese_categories.json \
        --weight_path ./checkpoints/viecap_vi_multilingual_v4/vietnamese-0024.pt \
        --out_path ./checkpoints/viecap_vi_multilingual_v4/ \
        --soft_prompt_first \
        --threshold 0.001


python src/compute_metrics.py --input ./checkpoints/viecap_vi_multilingual_v4/uit_vilc_generated_captions.json \
                              --output ./checkpoints/viecap_vi_multilingual_v4/metrics.json
