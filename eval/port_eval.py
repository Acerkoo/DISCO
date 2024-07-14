import json
import argparse
#import numpy as np

# from transformers import AutoTokenizer
tokenizer = None

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", required=True, type=str)
    parser.add_argument("--tokenizer_path", required=True, type=str)
    parser.add_argument("--filename", default="results.out", type=str)
    return parser.parse_args()

def obtain_f1_and_em_and_acc(pred, label, acc):
    if isinstance(pred, list):
        pred, label = pred[0], label[0]
    
    pred, label = pred.lower(), label.lower()
    if isinstance(acc, list):
        acc = acc[0]
    
    return 0, 0, acc

def calc_overlap(ids_preds, ids_labels, probs_preds=None, ground_ids=None):
    common_ids = (set(ids_preds) & set(ids_labels)) - (set(ids_labels) & set(ground_ids))
    overlap_list = [(v in common_ids) for v in ids_preds]
    overlap_rate = sum(overlap_list) / len(overlap_list)

    return overlap_rate, None, None


def avg(a):
    return round(sum(a) / len(a) * 100, 2)

def main(args):
    # global tokenizer
    # print("loading tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # print("loading tokenizer done...")

    input_file = args.res_dir + "/results.json"
    output_file = args.res_dir + "/" + args.filename

    with open(input_file, "r", encoding="utf-8") as fin:
        data_list = json.load(fin)
    
    portability_acc_list = []
    overlap_post_rate_list, overlap_cross_rate_list = [], []

    for item in data_list:
        _, _, portability_acc = obtain_f1_and_em_and_acc(
            item["post"]["portability"]["one_hop_acc"]["ans"],
            item["post"]["portability"]["one_hop_acc"]["target"],
            # None,
            item["post"]["portability"]["one_hop_acc"]["acc"],
        )
        portability_acc_list.append(portability_acc)

        targ_key = "targ_ids" if "targ_ids" in item["post"]["portability"]["one_hop_acc"] else "target_ids"

        overlap_post_rate, _, _ = calc_overlap(
            item["post"]["portability"]["one_hop_acc"]["ans_ids"],
            item["post"]["rewrite_acc"][targ_key],
            # item["post"]["portability"]["one_hop_acc"]["ans_prob"],
            None,
            item["post"]["portability"]["one_hop_acc"][targ_key],
        )
        overlap_post_rate_list.append(overlap_post_rate)

        overlap_cross_rate, _, _ = calc_overlap(
            item["post"]["portability"]["one_hop_acc"]["ans_ids"],
            item["pre"]["portability"]["one_hop_acc"]["ans_ids"],
            # item["post"]["portability"]["one_hop_acc"]["ans_prob"],
            None,
            item["post"]["portability"]["one_hop_acc"][targ_key],
        )
        overlap_cross_rate_list.append(overlap_cross_rate)

    print(f"Outdated issue = {avg(overlap_cross_rate_list)}% / Factual issue = {avg(overlap_post_rate_list)}%.")
    
    with open(output_file, "a+") as fout:
        fout.write(f"Outdated issue = {avg(overlap_cross_rate_list)}% / Factual issue = {avg(overlap_post_rate_list)}%.")

if __name__ == '__main__':
    main(parser_args())
