import os.path
# import sys
# sys.path.append('..')
import torch
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    DISCOHyperParams,
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from easyeditor.models.disco import encode_disco_facts
from sentence_transformers import SentenceTransformer
from easyeditor import CounterFactDataset

import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--save_dir', default='./output', type=str)
    return parser.parse_args()

def main(args):
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'DISCO':
        editing_hparams = DISCOHyperParams
    else:
        raise NotImplementedError

    test_data = json.load(open(args.data_file, 'r', encoding='utf-8'))

    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)

    prompts = [test_data_['requested_rewrite']["prompt"].format(test_data_['requested_rewrite']["subject"]) for test_data_ in test_data]
    rephrase_prompts = [edit_data_['generation_prompts'][0] for edit_data_ in test_data]
    target_new = [edit_data_['requested_rewrite']["target_new"]["str"] for edit_data_ in test_data]
    locality_prompts = [edit_data_['neighborhood_prompts'][0] for edit_data_ in test_data]
    locality_ans = [edit_data_['requested_rewrite']["target_true"]["str"] for edit_data_ in test_data]
    try:
        portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
        portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]
    except:
        portability_prompts, portability_ans = None, None

    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    portability_inputs = None
    if portability_prompts is not None:
        portability_inputs = {
            'one_hop':{
                'prompt': portability_prompts,
                'ground_truth': portability_ans,
            },
        }
    subject = [edit_data_['requested_rewrite']['subject'] for edit_data_ in test_data]
    hparams = editing_hparams.from_hparams(args.hparams_dir)

    if hasattr(hparams, "results_dir"):
        hparams.results_dir = args.save_dir
    
    if hasattr(hparams, 'alpha'):
        hparams.alpha = args.alpha
    
    if hasattr(hparams, "stats_dir"):
        hparams.stats_dir = args.save_dir

    if args.editing_method == 'IKE' or args.editing_method == "DISCO":
        train_data_path = args.train_file
        train_ds = CounterFactDataset(train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        if args.editing_method == "IKE":
            encode_ike_facts(sentence_model, train_ds, hparams)
        else:
            encode_disco_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    json.dump(metrics, open(os.path.join(args.save_dir, f'results.json'), 'w'), indent=4)

if __name__ == "__main__":
    main(parser_args())
