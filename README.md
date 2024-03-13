# TDD
Unveiling and Manipulating Prompt Influence in Large Language Models (ICLR 2024)

TDD explores using **token distributions** to explain **autoregressive LLMs**. 
Our another work, PromptExplainer, explains **masked language models such as BERT and RoBERTa using token distributions**. Welcome to check [PromptExplainer](https://github.com/zijian678/PromptExplainer)!

## Reproduce our results
The are two steps to reproduce our results.
* Step 1: Generate saliency scores using TDD_step1.py. You may choose different datasets and LLMs to generate saliency scores.
* Step 2: Evaluate using AOPC and Sufficiency by TDD_step2.py. It calculates AOPC and Suff scores using the saliency scores from step 1.

Please use your own LLaMA access token while experimenting with it.

## Acknowledgement
The code for contrastive explanation baselines is from [interpret-lm](https://github.com/kayoyin/interpret-lm). The dataset is from [BLiMP](https://github.com/alexwarstadt/blimp). We thank the authors for their excellent contributions!
