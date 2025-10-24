#!/bin/bash

EXP_NAME="viecap_vi_multilingual_mask20"
mkdir -p logs/$EXP_NAME
LOG_FILE=logs/$EXP_NAME/$(date "+%Y-%m-%d-%H-%M-%S").log

echo "=========================================================="
echo "TRAINING: $EXP_NAME"
echo "=========================================================="

python main.py \
--bs 64 \
--lr 1e-5 \
--epochs 20 \
--device cuda:0 \
--clip_model "BAAI/AltCLIP-m18" \
--using_clip_features \
--language_model NlpHUST/gpt2-vietnamese \
--random_mask \
--prob_of_random_mask 0.2 \
--using_hard_prompt \
--soft_prompt_first \
--path_of_datasets ./dataset/UIT-ViIC/uitviic_captions_train_with_features.pickle \
--name_of_objects_vocabs vietnamese_entities \
--path_of_objects_vocabs ./dataset/vietnamese_categories.json \
--out_dir ./checkpoints/$EXP_NAME \
--use_amp \
|& tee -a ${LOG_FILE}



EXP_NAME="viecap_vi_multilingual_mask40"
mkdir -p logs/$EXP_NAME
LOG_FILE=logs/$EXP_NAME/$(date "+%Y-%m-%d-%H-%M-%S").log

echo "=========================================================="
echo "TRAINING: $EXP_NAME"
echo "=========================================================="

python main.py \
--bs 64 \
--lr 1e-5 \
--epochs 20 \
--device cuda:0 \
--clip_model "BAAI/AltCLIP-m18" \
--using_clip_features \
--language_model NlpHUST/gpt2-vietnamese \
--random_mask \
--prob_of_random_mask 0.4 \
--using_hard_prompt \
--soft_prompt_first \
--path_of_datasets ./dataset/UIT-ViIC/uitviic_captions_train_with_features.pickle \
--name_of_objects_vocabs vietnamese_entities \
--path_of_objects_vocabs ./dataset/vietnamese_categories.json \
--out_dir ./checkpoints/$EXP_NAME \
--use_amp \
|& tee -a ${LOG_FILE}


EXP_NAME="viecap_vi_multilingual_mask60"
mkdir -p logs/$EXP_NAME
LOG_FILE=logs/$EXP_NAME/$(date "+%Y-%m-%d-%H-%M-%S").log

echo "=========================================================="
echo "TRAINING: $EXP_NAME"
echo "=========================================================="

python main.py \
--bs 64 \
--lr 1e-5 \
--epochs 20 \
--device cuda:0 \
--clip_model "BAAI/AltCLIP-m18" \
--using_clip_features \
--language_model NlpHUST/gpt2-vietnamese \
--random_mask \
--prob_of_random_mask 0.6 \
--using_hard_prompt \
--soft_prompt_first \
--path_of_datasets ./dataset/UIT-ViIC/uitviic_captions_train_with_features.pickle \
--name_of_objects_vocabs vietnamese_entities \
--path_of_objects_vocabs ./dataset/vietnamese_categories.json \
--out_dir ./checkpoints/$EXP_NAME \
--use_amp \
|& tee -a ${LOG_FILE}


EXP_NAME="viecap_vi_multilingual_mask80"
mkdir -p logs/$EXP_NAME
LOG_FILE=logs/$EXP_NAME/$(date "+%Y-%m-%d-%H-%M-%S").log

echo "=========================================================="
echo "TRAINING: $EXP_NAME"
echo "=========================================================="

python main.py \
--bs 64 \
--lr 1e-5 \
--epochs 20 \
--device cuda:0 \
--clip_model "BAAI/AltCLIP-m18" \
--using_clip_features \
--language_model NlpHUST/gpt2-vietnamese \
--random_mask \
--prob_of_random_mask 0.8 \
--using_hard_prompt \
--soft_prompt_first \
--path_of_datasets ./dataset/UIT-ViIC/uitviic_captions_train_with_features.pickle \
--name_of_objects_vocabs vietnamese_entities \
--path_of_objects_vocabs ./dataset/vietnamese_categories.json \
--out_dir ./checkpoints/$EXP_NAME \
--use_amp \
|& tee -a ${LOG_FILE}