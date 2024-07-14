from transformers import AutoTokenizer
from ..util import HyperParams
from typing import List
import typing
import torch
import numpy as np
from .evaluate_utils import  test_batch_prediction_acc, test_seq2seq_batch_prediction_acc, test_prediction_acc


def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: str,
    ground_truth: str,
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        textual_ans, textual_target, portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
    else:
        textual_ans, textual_target, ans_ids, target_ids, portability_correct, ans_probs, targ_probs = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device)

    ret = {
        f"{portability_key}_acc": {
            "ans": textual_ans,
            "target": textual_target,
            "acc": portability_correct,
            "ans_ids": ans_ids,
            "ans_prob": ans_probs,
            "target_ids": target_ids,
            "target_prob": targ_probs,
        }
    }
    return ret
