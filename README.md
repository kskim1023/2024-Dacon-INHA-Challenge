# 2024-Dacon-INHA-Challenge
2024 인하 인공지능 챌린지 - 경제기사분석 LLM 
2024 INHA AI Challenge - Economic News Analyzing LLM

The model is intended to analyze Korean Economy News and train(finetune) a LLM for QA tasks. The goal of the project is to find the most efficient finetuning method within our GPU environment by testing multiple LLMs and finetuning techniques including LLaMA3-openko, QLoRA and more.

Team: 요아소비빠다정(0.806,, 20th)

Information about the challenge:
https://dacon.io/competitions/official/236291/overview/rules


## Enviroment
- OS: Linux
- Python Version: 3.10
- GPU: A100 or RTX6000

## Models & Approaches
- Models
  - POLAR-14B
  - LLaMA3
  - LLaMA3-openko
    
- Approaches
  - Preprocessing & summarizing context with Sentence BERT and our own similarity algorithm
  - Jaccard + Cosine Similarity(6:4 ratio)
  - Finetuning with QLoRA
  - Prefix Tuning


## Running the Code
