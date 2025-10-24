import json
from collections import defaultdict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


def compute_metrics(result_json_path: str, output_path: str = "metrics.json"):
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gts = defaultdict(list)
    res = {}

    for item in data:
        image_name = item["image_name"]
        captions = item["captions"]
        prediction = item["prediction"].strip()

        gts[image_name] = captions
        res[image_name] = [prediction]

    scorers = [
        (Bleu(4), ["B@1", "B@2", "B@3", "B@4"]),
        (Meteor(), "M"),
        (Rouge(), "R"),
        (Cider(), "C"),
        (Spice(), "S"),
    ]

    final_scores = {}

    for scorer, method in scorers:
        print(f"Computing {method} ...")
        score, _ = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for m, s in zip(method, score):
                final_scores[m] = round(s * 100, 2)
        else:
            final_scores[method] = round(score * 100, 2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_scores, f, indent=4, ensure_ascii=False)

    print("\n=== Final Scores ===")
    print(json.dumps(final_scores, indent=4, ensure_ascii=False))

    return final_scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to eval result JSON file")
    parser.add_argument("--output", type=str, default="metrics.json", help="Output path for metrics JSON")
    args = parser.parse_args()

    compute_metrics(args.input, args.output)
