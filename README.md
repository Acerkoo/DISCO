# Outdated Issue Aware Decoding for Reasoning Questions on Edited Knowledge

ACL 2024 Findings: Outdated Issue Aware Decoding for Reasoning Questions on Edited Knowledge, paper in [arxiv](https://arxiv.org/abs/2406.02882).

Zengkui Sun, Yijin Liu, Jiaan Wang, Fandong Meng, Jinan Xu, Yufeng Chen$^{\dagger}$, Jie Zhou

Our code follows [EasyEdit](https://github.com/zjunlp/EasyEdit).

## Requirements

Detailed information in requirements.txt, and the packages can be installed as follows:

```
pip install -r requirements.txt
```




## Data

Data can be found in [EasyEdit](https://github.com/zjunlp/EasyEdit), and the direct data preprocessed by us can be found in [Google Drive](https://drive.google.com/file/d/1hHDXjCi78r3ksF87-ecJKHHmk6mvJi3s/view?usp=drive_link) (unzip in the `data` directory).



## Models

We use the `gpt-j-6b`, `llama-2-7b`, and `llama-2-13b` as the LLMs in our paper.

Note that, you can modify the path of pretrained model by setting the value of `plm_dir` in our scripts.



## Training / Inference

We display the scripts in `scripts`.

For example, you can run DISCO (ours) with `gpt-j-6b` as follows:

```
bash scripts/zsre_disco.sh
```

After the whole set tested, results will be saved in `$ckpt_dir/`, which could be modified in scripts.



## Evaluation

We test the `F1 / EM` with `eval/eval.sh`, `OE / TE` with `eval/port_eval.sh`, and the overall example can be found in `eval/gen_metrics.sh`.



## BibTex

If you find this repo useful for your research, please consider citing our paper:


```
@article{sun2024outdated,
  title={Outdated Issue Aware Decoding for Factual Knowledge Editing},
  author={Sun, Zengkui and Liu, Yijin and Wang, Jiaan and Meng, Fandong and Xu, Jinan and Chen, Yufeng and Zhou, Jie},
  journal={arXiv preprint arXiv:2406.02882},
  year={2024}
}
```