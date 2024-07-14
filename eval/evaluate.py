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

def obtain_f1_and_em_and_acc(pred, label, acc, pred_tokens=None, label_tokens=None):
    if isinstance(pred, list):
        pred, label = pred[0], label[0]
    
    pred, label = pred.lower(), label.lower()
    if isinstance(acc, list):
        acc = acc[0]
    
    if len(pred) == 0 and len(label) == 0:
        return 1.0, 1, acc
    if len(pred) == 0 or len(label) == 0:
        return 0.0, 0, acc

    # pred_tokens = tokenizer.encode(pred, add_special_tokens=False)
    # label_tokens = tokenizer.encode(label, add_special_tokens=False)
    
    em = 1 if pred == label else 0

    k = len(pred_tokens) * len(label_tokens)

    intersecting_words = []
    for word in pred_tokens.copy():
        if word in label_tokens:
            pred_tokens.remove(word)
            label_tokens.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em, acc

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
    
    reliability_f1_list, reliability_em_list, reliability_acc_list = [], [], []
    generalization_f1_list, generalization_em_list, generalization_acc_list = [], [], []
    locality_f1_list, locality_em_list, locality_acc_list = [], [], []
    locality_f1_golden_list, locality_em_golden_list, locality_acc_golden_list = [], [], []
    portability_f1_list, portability_em_list, portability_acc_list = [], [], []

    # reliability_pre_acc_list, generalization_pre_acc_list, portability_pre_acc_list = [], [], []

    for item in data_list:
        reliability_f1, reliability_em, reliability_acc = obtain_f1_and_em_and_acc(
            item["post"]["rewrite_acc"]["ans"],
            item["post"]["rewrite_acc"]["target"],
            item["post"]["rewrite_acc"]["acc"],
            item["post"]["rewrite_acc"]["ans_ids"],
            item["post"]["rewrite_acc"]["target_ids"],
            # item["pre"]["rewrite_acc"]["ans"],
        )
        reliability_f1_list.append(reliability_f1)
        reliability_em_list.append(reliability_em)
        reliability_acc_list.append(reliability_acc)
        # reliability_pre_acc_list.append(reliability_pre_acc)

        generalization_f1, generalization_em, generalization_acc = obtain_f1_and_em_and_acc(
            item["post"]["rephrase_acc"]["ans"],
            item["post"]["rephrase_acc"]["target"],
            item["post"]["rephrase_acc"]["acc"],
            item["post"]["rephrase_acc"]["ans_ids"],
            item["post"]["rephrase_acc"]["target_ids"],
            # item["pre"]["rephrase_acc"]["ans"],
        )
        generalization_f1_list.append(generalization_f1)
        generalization_em_list.append(generalization_em)
        generalization_acc_list.append(generalization_acc)

        if "locality" in item["post"] and "neighborhood_acc" in item["post"]["locality"] and 'post_ids' in item['post']['locality']['neighborhood_acc']:
            locality_f1, locality_em, locality_acc = obtain_f1_and_em_and_acc(
                item["post"]["locality"]["neighborhood_acc"]["post"],
                item["post"]["locality"]["neighborhood_acc"]["pre"],
                item["post"]["locality"]["neighborhood_acc"]["acc"],
                item["post"]["locality"]["neighborhood_acc"]["post_ids"],
                item["post"]["locality"]["neighborhood_acc"]["pre_ids"],
            )
            locality_f1_list.append(locality_f1)
            locality_em_list.append(locality_em)
            locality_acc_list.append(locality_acc)

            # locality_f1_golden, locality_em_golden, locality_acc_golden = obtain_f1_and_em_and_acc(
            #     item["post"]["locality"]["neighborhood_acc"]["post_ids"],
            #     item["post"]["locality"]["neighborhood_acc"]["golden_ids"],
            #     item["post"]["locality"]["neighborhood_acc"]["acc"],
            #     item["post"]["locality"]["neighborhood_acc"]["post_ids"],
            #     item["post"]["locality"]["neighborhood_acc"]["golden_ids"],
            # )
            # locality_f1_golden_list.append(locality_f1_golden)
            # locality_em_golden_list.append(locality_em_golden)
            # locality_acc_golden_list.append(locality_acc_golden)



        portability_f1, portability_em, portability_acc = obtain_f1_and_em_and_acc(
            item["post"]["portability"]["one_hop_acc"]["ans"],
            item["post"]["portability"]["one_hop_acc"]["target"],
            item["post"]["portability"]["one_hop_acc"]["acc"],
            item["post"]["portability"]["one_hop_acc"]["ans_ids"],
            item["post"]["portability"]["one_hop_acc"]["target_ids"],
            # item["pre"]["portability"]["one_hop_acc"]["ans"],
        )
        portability_f1_list.append(portability_f1)
        portability_em_list.append(portability_em)
        portability_acc_list.append(portability_acc)
        # portability_pre_acc_list.append(portability_pre_acc)

    # # print(portability_f1_list[:5])
    # # print(portability_em_list[:5])
    # # print(portability_acc_list[:5])
    print("-"* 5 + "F1 / EM / Acc score" + "-" * 5)
    print(f"reliablity: {avg(reliability_f1_list)} / {avg(reliability_em_list)} / {avg(reliability_acc_list)}.")
    print(f"generalization: {avg(generalization_f1_list)} / {avg(generalization_em_list)} / {avg(generalization_acc_list)}.")
    if len(locality_acc_list) > 0:
        print(f"locality: {avg(locality_f1_list)} / {avg(locality_em_list)} / {avg(locality_acc_list)}.")
        # print(f"locality_golden: {avg(locality_f1_golden_list)} / {avg(locality_em_golden_list)} / {avg(locality_acc_list)}.")
    print(f"portability: {avg(portability_f1_list)} / {avg(portability_em_list)} / {avg(portability_acc_list)}.")
    
    with open(output_file, "w") as fout:
        fout.write("-"* 5 + "F1 / EM / Acc score" + "-" * 5 + "\n")
        fout.write(f"reliablity: {avg(reliability_acc_list)} / {avg(reliability_em_list)} / {avg(reliability_acc_list)}.\n")
        fout.write(f"generalization: {avg(generalization_acc_list)} / {avg(generalization_em_list)} / {avg(generalization_acc_list)}.\n")
        if len(locality_acc_list) > 0:
            fout.write(f"locality: {avg(locality_f1_list)} / {avg(locality_em_list)} / {avg(locality_acc_list)}.\n")
        fout.write(f"portability: {avg(portability_f1_list)} / {avg(portability_em_list)} / {avg(portability_acc_list)}.\n")

if __name__ == '__main__':
    main(parser_args())
