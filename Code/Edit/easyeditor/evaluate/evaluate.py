"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

from shutil import register_unpack_format
import typing
from itertools import chain
from typing import List, Optional
from xmlrpc.client import boolean

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality
from .evaluate_utils import test_seq2seq_batch_prediction_acc, test_batch_prediction_acc, \
                            test_prediction_acc,test_generation_quality, PPL

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(
        model, model_name, hparams, tok, 
        rewrite_prompts, 
        target_new, 
        device=device, 
        eval_metric=eval_metric
    )

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(
                model, model_name, hparams, tok, 
                rephrase_prompts, 
                target_new, 
                device=device, 
                test_rephrase=True, 
                eval_metric=eval_metric
            )
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(
                    model, model_name, hparams, tok, 
                    locality_key, 
                    record['locality'][locality_key]['prompt'], 
                    record['locality'][locality_key]['ground_truth'], 
                    device=device
                )
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(
                    model, model_name, hparams, tok, 
                    portability_key, 
                    record['portability'][portability_key]['prompt'], 
                    record['portability'][portability_key]['ground_truth'], 
                    device=device
                )
            )
    if  test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts \
                        if isinstance(rewrite_prompts,list) \
                        else [rewrite_prompts,], max_out_len=100)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    if eval_metric == 'ppl':
        ppl = PPL(model, tok, prompt, target_new, device)
        ret = {
            f"{key}_ppl": ppl
        }
    else:
        if 't5' in model_name.lower():
            textual_ans, textual_target, acc = test_seq2seq_batch_prediction_acc(
                model, tok, hparams, 
                prompt, 
                target_new, 
                device
            )
        else:
            textual_ans, textual_target, ans_ids, target_ids, acc, ans_probs, targ_probs = test_prediction_acc(
                model, tok, hparams, 
                prompt, 
                target_new, 
                device
            )
        ret = {
            f"{key}_acc": {
                "ans": textual_ans,
                "target": textual_target,
                "acc": acc,
                "ans_ids": ans_ids,
                "ans_prob": ans_probs,
                "target_ids": target_ids,
                "target_prob": targ_probs,
            }
        }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: str,
    locality_ground_truth: str,
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        textual_ans = test_seq2seq_batch_prediction_acc(
            model, tok, hparams, 
            prompt, 
            locality_ground_truth, 
            device, 
            locality=True
        )
    else:
        ans, textual_ans, ans_probs = test_prediction_acc(
            model, tok, hparams, 
            prompt, 
            locality_ground_truth, 
            device, 
            locality=True
        )

    # if type(textual_ans) is not list:
    #     textual_ans = [textual_ans,]

    ret = {
        f"{locality_key}_output": {
            "ans": ans,
            "textual": textual_ans,
            "prob": ans_probs,
            # "target": textual_target,
        } 
    }
    # print(ret
    return ret

def compute_icl_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    icl_func,
    # icl_examples,
    record: typing.Dict,
    device,
    train_ds=None,
    pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    edit_fact = f"New Fact: {prompt} {target_new}\n"
    query_prompt = edit_fact + "Prompt: " + prompt
    if pre_edit:
        icl_examples = [""]
    else:
        icl_examples = icl_func(
            model,
            tok,
            record,
            hparams,
            query_prompt,
            copy=False,
            return_orig_weight=True,
            keep_original_weight=True,
            train_ds=train_ds,
        )

    # if pre_edit:
    textual_ans, textual_target, ans_ids, target_ids, edit_acc, ans_probs, target_probs, pre_logits = icl_lm_eval(
        model, model_name, hparams, tok, 
        [""], 
        target_new, 
        prompt
    )
    new_fact_ids = target_ids

    if not pre_edit:
    # else:
        pre_ans_ids = ans_ids
        textual_ans, textual_target, ans_ids, target_ids, edit_acc, ans_probs, target_probs, _ = icl_lm_eval(
            model, model_name, hparams, tok, 
            icl_examples, 
            target_new, 
            edit_fact + f"First Prompt: {rephrase} {target_new}\n" + f"Second Prompt: {prompt}",
            pre_logits=pre_logits,
            pre_ans_ids=pre_ans_ids,
            new_fact_ids=new_fact_ids,
        )

    # rew_ans = textual_ans.copy()
    ret = {
        "rewrite_acc": {
            "ans": textual_ans,
            "target": textual_target,
            "acc": edit_acc,
            "ans_ids": ans_ids,
            "ans_prob": ans_probs, 
            "target_ids": target_ids,
            "targ_prob": target_probs,
        }
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        # if pre_edit:
        textual_ans, textual_target, ans_ids, target_ids, rephrase_acc, ans_probs, target_probs, pre_logits = icl_lm_eval(
            model, model_name, hparams, tok, 
            [""], 
            target_new, 
            rephrase,
        )
            
        if not pre_edit:
            textual_ans, textual_target, ans_ids, target_ids, rephrase_acc, ans_probs, target_probs, _ = icl_lm_eval(
                model, model_name, hparams, tok, 
                icl_examples, 
                target_new, 
                edit_fact + f"First Prompt: {rephrase} {target_new}\n" + f"Second Prompt: {rephrase}",
                pre_logits=pre_logits,
                pre_ans_ids=pre_ans_ids,
                new_fact_ids=new_fact_ids,
            )

        ret['rephrase_acc'] = {
            "ans": textual_ans,
            "target": textual_target,
            "acc": rephrase_acc,
            "ans_ids": ans_ids,
            "ans_prob": ans_probs, 
            "target_ids": target_ids,
            "targ_prob": target_probs,
        }
    
    if 'locality' in record.keys() and any(record['locality']) and not pre_edit:
        for locality_key in record['locality'].keys():
            pre_ids, pre_neighbor, pre_probs, pre_logits, _, _ = icl_lm_eval(
                model, model_name, hparams, tok, 
                [''], 
                record['locality'][locality_key]['ground_truth'], 
                record['locality'][locality_key]['prompt'],
                neighborhood=True, 
            )

            post_ids, post_neighbor, post_probs, _, golden_ids, golden_probs = icl_lm_eval(
                model, model_name, hparams, tok, 
                icl_examples, 
                record['locality'][locality_key]['ground_truth'],
                edit_fact + f"First Prompt: {rephrase} {target_new}\n" + f"Second Prompt: {record['locality'][locality_key]['prompt']}",
                neighborhood=True,
                pre_logits=pre_logits,
                pre_ans_ids=pre_ids,
                new_fact_ids=new_fact_ids,
            )

            if type(pre_neighbor) is not list:
                pre_neighbor = [pre_neighbor, ]
            if type(post_neighbor) is not list:
                post_neighbor = [post_neighbor, ]
            assert len(pre_neighbor) == len(post_neighbor)

            acc = np.mean(np.equal(pre_ids, post_ids))
            ret['locality'][f'{locality_key}_acc'] = {
                "post": post_neighbor,
                "pre": pre_neighbor,
                "acc": acc,
                "post_ids": post_ids,
                "post_prob": post_probs,
                "pre_ids": pre_ids,
                "pre_prob": pre_probs,
                "golden_ids": golden_ids,
                "golden_probs": golden_probs,
            }

    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            # if pre_edit:
            textual_ans, textual_target, ans_ids, target_ids, portability_acc, ans_probs, target_probs, pre_logits = icl_lm_eval(
                model, model_name, hparams, tok, 
                [""],
                record['portability'][portability_key]['ground_truth'], 
                record['portability'][portability_key]['prompt'],
            )

            if not pre_edit:
                pre_ans_ids = ans_ids

                second_question = record['portability'][portability_key]['prompt']
                history = f"History: {second_question} {textual_ans}\n"
                edit_fact = f"New Fact: {prompt} {target_new}\n"
                sec_query_prompt = edit_fact + "Prompt: " + second_question

                port_icl_examples = icl_func(
                    model,
                    tok,
                    record,
                    hparams,
                    sec_query_prompt,
                    copy=False,
                    return_orig_weight=True,
                    keep_original_weight=True,
                    train_ds=train_ds,
                )

                query_prompt = edit_fact + f"First Prompt: {rephrase} {target_new}\n" + "Second Prompt: " + record['portability'][portability_key]['prompt']

                textual_ans, textual_target, ans_ids, target_ids, portability_acc, ans_probs, target_probs, _ = icl_lm_eval(
                    model, model_name, hparams, tok, 
                    port_icl_examples, 
                    record['portability'][portability_key]['ground_truth'], 
                    query_prompt,
                    pre_logits=pre_logits,
                    pre_ans_ids=pre_ans_ids,
                    new_fact_ids=new_fact_ids,
                )

            ret['portability'][f'{portability_key}_acc'] = {
                "ans": textual_ans,
                "target": textual_target,
                "acc": portability_acc,
                "ans_ids": ans_ids,
                "ans_prob": ans_probs, 
                "target_ids": target_ids,
                "targ_prob": target_probs,
            }

    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False,
        portability=None,
        pre_logits=None,
        new_fact_ids=None,
        pre_ans_ids=None,
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits

            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            
            ans_idss = ans.detach().cpu().numpy().tolist()
            target_idss = target_ids.detach().cpu().squeeze().numpy().tolist()
            if not isinstance(ans_idss, list):
                ans_idss = [ans_idss]

            textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True).strip()
            textual_target = tokenizer.decode(target_idss, skip_special_tokens=True).strip()

            if neighborhood:
                return textual_ans

            return textual_ans, textual_target, \
                torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()  

    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        logits = logits[:, -target_ids.size(1): -1].cpu()

        logits = logits.softmax(dim=-1)
        if pre_logits is not None:
            mask = torch.zeros_like(logits) == 1

            up_rate_mask = (logits - pre_logits) > 0
            if pre_ans_ids is not None:
                pre_ans_ids = torch.tensor(pre_ans_ids).view(mask.shape[0], 1, -1).repeat(1, mask.shape[1], 1)
                mask = mask.scatter(-1, pre_ans_ids, True)

            if new_fact_ids is not None:
                new_fact_ids = torch.tensor(new_fact_ids).view(mask.shape[0], 1, -1).repeat(1, mask.shape[1], 1)
                mask = mask.scatter(-1, new_fact_ids, True)

            mask = (mask & up_rate_mask)
            mask = ~mask
            sub_weights = mask.float() * (logits - pre_logits) 
            
            logits = logits + hparams.alpha * sub_weights

        ans = torch.argmax(logits, dim=-1).squeeze()
        target_ids = target_ids[:,1:]   

        ans_idss = ans.detach().cpu().numpy().tolist()
        target_idss = target_ids.detach().cpu().squeeze().numpy().tolist()
        if not isinstance(ans_idss, list):
            ans_idss, target_idss = [ans_idss], [target_idss]

        ans_probs = [round(v, 4) for v in logits[0].gather(-1, torch.tensor(ans_idss).unsqueeze(-1)).squeeze(-1).tolist()]
        targ_probs = [round(v, 4) for v in logits[0].gather(-1, torch.tensor(target_idss).unsqueeze(-1)).squeeze(-1).tolist()]

        textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True).strip()
        textual_target = tokenizer.decode(target_idss, skip_special_tokens=True).strip()

        if neighborhood:
            return ans_idss, textual_ans, ans_probs, logits, target_idss, targ_probs

        return textual_ans, textual_target, \
            ans_idss, target_idss, \
            torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist(), \
            ans_probs, targ_probs, logits

    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        logits = logits[:, -target_ids.size(1): -1].cpu()
        logits = logits.softmax(dim=-1)

        if pre_logits is not None:
            mask = torch.zeros_like(logits) == 1

            up_rate_mask = (logits - pre_logits) > 0
            if pre_ans_ids is not None:
                pre_ans_ids = torch.tensor(pre_ans_ids).view(mask.shape[0], 1, -1).repeat(1, mask.shape[1], 1)
                mask = mask.scatter(-1, pre_ans_ids, True)

            if new_fact_ids is not None:
                new_fact_ids = torch.tensor(new_fact_ids).view(mask.shape[0], 1, -1).repeat(1, mask.shape[1], 1)
                mask = mask.scatter(-1, new_fact_ids, True)

            mask = (mask & up_rate_mask)
            mask = ~mask
            sub_weights = mask.float() * (logits - pre_logits) 
            
            logits = logits + hparams.alpha * sub_weights

        ans = torch.argmax(logits, dim=-1).squeeze()
        target_ids = target_ids[:,:-1]

        ans_idss = ans.detach().cpu().numpy().tolist()
        target_idss = target_ids.detach().cpu().squeeze().numpy().tolist()

        if not isinstance(ans_idss, list):
            ans_idss, target_idss = [ans_idss], [target_idss]
        
        ans_probs = [round(v, 4) for v in logits[0].gather(-1, torch.tensor(ans_idss).unsqueeze(-1)).squeeze(-1).tolist()]
        targ_probs = [round(v, 4) for v in logits[0].gather(-1, torch.tensor(target_idss).unsqueeze(-1)).squeeze(-1).tolist()]

        textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True).replace("\n", " ").strip()
        textual_target = tokenizer.decode(target_idss, skip_special_tokens=True).replace("\n", " ").strip()

        if neighborhood:
            return ans_idss, textual_ans, ans_probs, logits
        
        return textual_ans, textual_target, \
            ans_idss, target_idss, \
            torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist(), \
            ans_probs, targ_probs, logits

def compute_icl_multimodal_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    # vis_tok,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image
    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record["image"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    
    # assert "image" in record.keys() or print("IKE for multimodal needs image.")
    # image_path = [os.path.join(vis_root, i) for i in record["image"]]
    # rephrase_image_path = [os.path.join(rephrase_root, i) for i in record["image_rephrase"]] if "image_rephrase" in record.keys() else image_path
    # image, rephrase_image = ([vis_tok(Image.open(ip).convert("RGB")) for ip in x] for x in [image_path, rephrase_image_path]) 
    # image, rephrase_image = (record[x] for x in ["image", "image_rephrase"])
    
    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
        
    if "multimodal_locality_image" in record.keys():
        # m_loc_image_path = [os.path.join(vis_root, i) for i in record["m_loc"]]
        # m_loc_image = [vis_tok(Image.open(ip).convert("RGB")) for ip in m_loc_image_path]
        m_loc_image = record["multimodal_locality_image"]
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]
    
    new_fact = f'New Fact: {prompt} {target}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       target, prompt, image)
    else:
        edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              target, new_fact, image)
    ret = {
        f"rewrite_acc": edit_acc
    }

    if rephrase is not None:
        rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
        ret['rephrase_acc'] = rephrase_acc
        
    if "image_rephrase" in record.keys():
        rephrase_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target, new_fact, rephrase_image)
        ret['rephrase_image_acc'] = rephrase_image_acc
    
    if "locality_prompt" in record.keys():
        locality_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                loc_a, f'New Fact: {loc_q} {loc_a}\nPrompt: {loc_q}', None)
        ret['locality_acc'] = locality_acc
    
    if "multimodal_locality_image" in record.keys():
        locality_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               m_loc_a, f'New Fact: {m_loc_q} {m_loc_a}\nPrompt: {m_loc_q}', m_loc_image)
        ret['locality_image_acc'] = locality_image_acc
            
    return ret

def icl_multimodal_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        image,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    # if 't5' in model_name.lower():
    #     target_len = len(tokenizer.encode(target))
    #     target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
    #     encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
    #     input_ids = encodings['input_ids'].to(device)
    #     attention_mask = encodings['attention_mask'].to(device)
    #     with torch.no_grad():
    #         logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
    #         ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
    #         target_ids = target_ids[:,-target_len:-1]
    #         if neighborhood:
    #             return ans.squeeze().detach().cpu().numpy().tolist()
    #         return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()

    # if image is not None and len(image.shape) == 3:
    #     image = image.unsqueeze(0)
    # samples = {}
    # samples['text_input'] = [''.join(icl_examples) + f'{x} {target}']
    # samples['image'] = image
    # if hasattr(model, 'llama_model'):
    #     samples['prompts_len'] = [len(tokenizer.encode(''.join(icl_examples) + f'{x}', add_special_tokens=False))]
    # else:
    #     samples['prompts_len'] = [len(tokenizer.encode(''.join(icl_examples) + f'{x}'))]
    samples = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image) 
    # if logits.dim() == 3:
    #     logits = logits[:, :-1]
    #     targ = labels[:, 1:]
    #     logits = logits[:, -targ.size(1):]
    # mask = targ != -100
    # targ[~mask] = 0
    # pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
    # correct = pred_ids == targ
    # correct = correct & mask
    # num_non_padding = mask.sum().float().item()
    # acc = correct.sum() / num_non_padding
    
    return compute_multimodal_edit_quality(model, samples)

def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target,]
    if isinstance(prompts, str):
        prompts = [prompts,]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]
    
    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt,)) for prompt in prompts]  
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len        
    } 
    return ret

def compute_multimodal_edit_quality(model, batch):
    
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy()
  
def compute_multimodal_edit_quality_demo(model, batch):
    
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy(), logits

def compute_multimodal_edit_results(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret
  
def compute_multimodal_edit_results_demo(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _, logits = compute_multimodal_edit_quality_demo(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret, logits


    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]
