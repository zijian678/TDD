# TDD
**Unveiling and Manipulating Prompt Influence in Large Language Models (ICLR 2024)** [Link] (https://iclr.cc/virtual/2024/poster/18355)

TDD explores using **token distributions** to explain **autoregressive LLMs**. 
Our another work, PromptExplainer, explains **masked language models such as BERT and RoBERTa using token distributions**. Welcome to check [PromptExplainer](https://github.com/zijian678/PromptExplainer)!

## Reproduce our results
The are two steps to reproduce our results.
* Step 1: Generate saliency scores using TDD_step1.py. You may choose different datasets and LLMs to generate saliency scores.
* Step 2: Evaluate using AOPC and Sufficiency by TDD_step2.py. It calculates AOPC and Suff scores using the saliency scores from step 1.

Please use your own LLaMA access token while experimenting with it.

## Acknowledgement
The code for contrastive explanation baselines is from [interpret-lm](https://github.com/kayoyin/interpret-lm). The dataset is from [BLiMP](https://github.com/alexwarstadt/blimp). We thank the authors for their excellent contributions!

## Citation
If you find our work useful, please consider citing TDD:
```
@inproceedings{feng2024tdd,
  title={Unveiling and Manipulating Prompt Influence in Large Language Models},
  author={Feng, Zijian and Zhou, Hanzhang and Zhu, Zixiao and Qian, Junlang and Mao, Kezhi},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
