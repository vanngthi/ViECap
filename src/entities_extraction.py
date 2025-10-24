import os
import json
import pickle
import argparse
from tqdm import tqdm
from typing import List
from underthesea import pos_tag
from load_annotations import load_captions


def extract_entities(captions: str, vncorenlp_model, phonlp_model, path: str) -> List[str]:
    new_captions = []
    for caption in tqdm(captions):
        detected_entities, temp = [], []
        segmented = vncorenlp_model.word_segment(caption)[0]
        tokens, pos_tags, ner_tags, deps = phonlp_model.annotate(text=segmented)
        tokens = tokens[0]
        pos_tags = [p[0] for p in pos_tags[0]]
        ner_tags = ner_tags[0]
        deps = deps[0]

        raw_entities = []
        for i, (tok, pos, ner, dep) in enumerate(zip(tokens, pos_tags, ner_tags, deps)):
            head_idx, dep_label = dep
            if (
                pos in {"N", "Nc", "Np"} or 
                dep_label in {"nmod", "sub", "dob", "pob"} or
                ner != "O"
            ):
                raw_entities.append(tok.strip().lower())

        merged_entities, temp = [], []
        for i, pos in enumerate(pos_tags):
            if pos in {"N", "Nc", "Np"}:
                temp.append(tokens[i].lower())
            else:
                if temp:
                    merged_entities.append(" ".join(temp))
                    temp = []
        if temp:
            merged_entities.append(" ".join(temp))

        # final_entities = list(dict.fromkeys(raw_entities + merged_entities))
        # final_entities = [
        #     e.replace("_", " ").strip() 
        #     for e in final_entities 
        # ]
        merged_entities = [m.replace("_", " ").strip() for m in merged_entities]
        filtered_raw = []
        for r in raw_entities:
            r_clean = r.replace("_", " ").strip()
            if not any(r_clean in m for m in merged_entities):
                filtered_raw.append(r_clean)

        final_entities = merged_entities + filtered_raw
        final_entities = list(dict.fromkeys(final_entities))

        new_captions.append([final_entities, caption])
        
    with open(path, 'wb') as outfile:
        pickle.dump(new_captions, outfile)


if __name__ == '__main__':
    print("Entities Extracting ... ...", flush=True)
    parser = argparse.ArgumentParser(description="Extract Vietnamese entities from caption dataset")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Tên dataset, ví dụ: viecap, viecap_test, flickr30k_vi, ...")
    parser.add_argument('--captions_path', type=str, required=True,
                        help="Đường dẫn tới file JSON caption")
    parser.add_argument('--out_path', type=str, required=True,
                        help="Đường dẫn file pickle đầu ra (.pkl)")

    args = parser.parse_args()
    
    if os.path.exists(args.out_path):
        print('Read!')
        with open(args.out_path, 'rb') as infile:
            captions_with_entities = pickle.load(infile)
        print(f'The length of datasets: {len(captions_with_entities)}')
        captions_with_entities = captions_with_entities[:20]
        for caption_with_entities in captions_with_entities:
            print(caption_with_entities)
    else:
        print('Writing... ...')
        import phonlp
        import py_vncorenlp

        # phonlp.download(save_dir='./nlp_models/phonlp')
        # py_vncorenlp.download_model(save_dir='./nlp_models/vncorenlp')

        jar_dir = os.path.abspath("./nlp_models/vncorenlp")
        pho_dir = os.path.abspath("./nlp_models/phonlp")
        captions_path = os.path.abspath(args.captions_path)
        out_path = os.path.abspath(args.out_path)
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
        os.environ["CLASSPATH"] = os.path.join(jar_dir, "VnCoreNLP-1.2.jar")

        vncorenlp_model = py_vncorenlp.VnCoreNLP(
            save_dir=jar_dir,
            annotators=["wseg"],
            max_heap_size='-Xmx2g'
        )

        phonlp_model = phonlp.load(pho_dir)
        captions = load_captions(args.dataset_name, captions_path)
        extract_entities(captions, vncorenlp_model, phonlp_model, out_path) 