#!/bin/bash

echo "Start Entities Extraction"
python src/entities_extraction.py --dataset_name uit_vilc \
                                  --captions_path ./dataset/UIT-ViIC/uitviic_captions_test2017.json \
                                  --out_path ./dataset/UIT-ViIC/uitviic_captions_test_with_entities.pickle

# # python src/entities_extraction.py --dataset_name uit_vilc \
# #                                   --captions_path ./dataset/UIT-ViIC/uitviic_captions_val2017.json \
# #                                   --out_path ./dataset/UIT-ViIC/uitviic_captions_val_with_entities.pickle

# python src/entities_extraction.py --dataset_name uit_vilc \
#                                   --captions_path ./dataset/UIT-ViIC/uitviic_captions_train2017.json \
#                                   --out_path ./dataset/UIT-ViIC/uitviic_captions_train_with_entities.pickle

# python src/entities_extraction.py --dataset_name flick_sportball \
#                                   --captions_path ./dataset/Flick_sportball/train.json \
#                                   --out_path ./dataset/Flick_sportballflick_captions_train_with_entities.pickle

python src/entities_extraction.py --dataset_name flick_sportball \
                                  --captions_path ./dataset/Flick_sportball/test.json \
                                  --out_path ./dataset/Flick_sportball/flick_captions_test_with_entities.pickle


echo "Extracting Text Features"
python src/texts_features_extraction.py --inpath ./dataset/UIT-ViIC/uitviic_captions_test_with_entities.pickle \
                                        --outpath ./dataset/UIT-ViIC/uitviic_captions_test_with_features.pickle

# # python src/texts_features_extraction.py --inpath ./dataset/UIT-ViIC/uitviic_captions_val_with_entities.pickle \
# #                                         --outpath ./dataset/UIT-ViIC/uitviic_captions_val_with_features.pickle

# python src/texts_features_extraction.py --inpath ./dataset/UIT-ViIC/uitviic_captions_train_with_entities.pickle \
#                                         --outpath ./dataset/UIT-ViIC/uitviic_captions_train_with_features.pickle

# python src/texts_features_extraction.py --inpath ./dataset/Flick_sportball/flick_captions_train_with_entities.pickle\
#                                         --outpath ./dataset/Flick_sportball/flick_captions_train_with_features.pickle

python src/texts_features_extraction.py --inpath ./dataset/Flick_sportball/flick_captions_test_with_entities.pickle\
                                        --outpath ./dataset/Flick_sportball/flick_captions_test_with_features.pickle


# echo "Extracting Images Features"
# python src/images_features_extraction.py