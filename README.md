# Rare Tokens Degenerate All Tokens: Improving Neural Text Generation via Adaptive Gradient Gating for Rare Token Embeddings
A Pytorch implementation of the paper:

[Rare Tokens Degenerate All Tokens: Improving Neural Text Generation via Adaptive Gradient Gating for Rare Token Embeddings](https://arxiv.org/abs/2109.03127) \
Sangwon Yu, Jongyoon Song, Heeseung Kim, Seong-min Lee, Woo-Jong Ryu, Sungroh Yoon

We present code for training models described in the paper, as well as pre-trained models.

# Setup
 - Require python version >= 3.6
 - The implementation is based on a [fairseq](https://github.com/pytorch/fairseq) 0.9.0.
 - Clone this repo and install requirements based on fairseq:

```
git clone https://github.com/ysw1021/AGG.git
cd AGG
pip install --editable .
```
      
# Data Processing
 ### WikiText-103 dataset for language modeling
 
  We preprocessed WikiText-103 dataset using GPT-2 tokenizer.
 
```
cd data/language_model
bash prepare-wikitext-103.sh
python gpt2_tokenize.py
```
The commands below assume you are in the ```$AGG``` directory.
```
python -u preprocess.py --only-source --trainpref data/language_model/wikitext-103/wiki.train.bpetokens \
--validpref data/language_model/wikitext-103/wiki.valid.bpetokens \
--testpref data/language_model/wikitext-103/wiki.test.bpetokens --destdir data-bin/wikitext-103_bpe --workers 20
```
 ### WMT'14 English to German dataset for neural machine translation
 
```
cd data/translation_wmt
bash prepare-wmt14en2de.sh --icml17
# or to use additional news-commentary-v12 data from WMT'17:
# bash prepare-wmt14en2de.sh
```
The commands below assume you are in the ```$AGG``` directory.
```
python -u preprocess.py --source-lang en --target-lang de \
--trainpref data/translation_wmt/wmt14_en_de/train \
--validpref data/translation_wmt/wmt14_en_de/valid \
--testpref data/translation_wmt/wmt14_en_de/test \
--destdir data-bin/wmt14_en_de_joined --workers 8 --joined-dictionary
```

# Training
 We tested scripts using an NVIDIA A40 gpu in a single setting. In multi-gpu settings, we found that there is a problem about batch data processing of AGG loss function: mini-batch samples in the gpus except the first gpu are ignored while calculating token appearance for rare token grouping. Solutions for multi-gpu settings will be updated later. 
 
 If you get OOM errors, try decreasing the batch size (```--max-tokens```,```--tokens-per-sample```).
 
 The commands below assume you are in the ```$AGG``` directory.
 
  ## Language Modeling with WikiText-103 dataset

   ### Baseline (MLE) Model
   
     python -u train.py --task language_modeling data-bin/wikitext-103_bpe --save-dir checkpoints/baseline_model \
     --tensorboard-logdir checkpoints/baseline_model --arch transformer_lm_gpt2_small --share-decoder-input-output-embed \
     --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --criterion cross_entropy --seed 1 --weight-decay 0.01 \
     --clip-norm 0.0 --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --no-progress-bar --log-interval 100 \
     --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --tokens-per-sample 1024 \
     --sample-break-mode none --max-tokens 4096 --update-freq 8 --fp16 --max-update 50000
     
   ### Train Model with AGG loss
   
     python -u train.py --task language_modeling data-bin/wikitext-103_bpe --save-dir checkpoints/agg_model \
     --tensorboard-logdir checkpoints/agg_model --arch transformer_lm_gpt2_small --share-decoder-input-output-embed \
     --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --criterion cross_entropy_agg --agg-alpha 0.03\
     --seed 1 --weight-decay 0.01 --clip-norm 0.0 --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --no-progress-bar \
     --log-interval 100 --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --tokens-per-sample 1024 \
     --sample-break-mode none --max-tokens 4096 --update-freq 8 --fp16 --max-update 50000
     
  ## Neural Machine Translation with WMT'14 English to German dataset
  
   ### Baseline (Transformer-base [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)) Model
   
     python -u train.py data-bin/wmt14_en_de_joined --save-dir checkpoints/translation_baseline_base \
     --tensorboard-logdir checkpoints/translation_baseline_base --ddp-backend=no_c10d --clip-norm 0.0 \
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --arch transformer_wmt_en_de \
     --share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --lr '1e-3' --lr-scheduler inverse_sqrt \
     --min-lr '1e-09' --warmup-updates 4000 --warmup-init-lr '1e-07' --weight-decay 0.01 --dropout 0.3 \
     --log-format 'simple' --log-interval 100 --max-tokens 8192 --save-interval-updates 2000 --max-update 200000 \
     --keep-interval-updates 10 --no-progress-bar --update-freq 8 --fp16 --seed 1
     
   ### Baseline (Transformer-big [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)) Model
   
     python -u train.py data-bin/wmt14_en_de_joined --save-dir checkpoints/translation_baseline_base \
     --tensorboard-logdir checkpoints/translation_baseline_base --ddp-backend=no_c10d --clip-norm 0.0 \
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --arch transformer_wmt_en_de_big \
     --share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --lr '1e-3' --lr-scheduler inverse_sqrt \
     --min-lr '1e-09' --warmup-updates 4000 --warmup-init-lr '1e-07' --weight-decay 0.01 --dropout 0.3 \
     --log-format 'simple' --log-interval 100 --max-tokens 8192 --save-interval-updates 2000 --max-update 200000 \
     --keep-interval-updates 10 --no-progress-bar --update-freq 8 --fp16 --seed 1
     
   ### Train Transformer-base Model with AGG loss
   
     python -u train.py data-bin/wmt14_en_de_joined --save-dir checkpoints/translation_agg_base \
     --tensorboard-logdir checkpoints/translation_agg_base --ddp-backend=no_c10d --clip-norm 0.0 \
     --criterion cross_entropy_agg --label-smoothing 0.1 --agg-alpha 0.08 --arch transformer_wmt_en_de \
     --share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --lr '1e-3' --lr-scheduler inverse_sqrt \
     --min-lr '1e-09' --warmup-updates 4000 --warmup-init-lr '1e-07' --weight-decay 0.01 --dropout 0.3 \
     --log-format 'simple' --log-interval 100 --max-tokens 8192 --save-interval-updates 2000 --max-update 200000 \
     --keep-interval-updates 10 --no-progress-bar --update-freq 8 --fp16 --seed 1
     
   ### Train Transformer-big Model with AGG loss
   
     python -u train.py data-bin/wmt14_en_de_joined --save-dir checkpoints/translation_agg_base \
     --tensorboard-logdir checkpoints/translation_agg_base --ddp-backend=no_c10d --clip-norm 0.0 \
     --criterion cross_entropy_agg --label-smoothing 0.1 --agg-alpha 0.08 --arch transformer_wmt_en_de_big \
     --share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --lr '1e-3' --lr-scheduler inverse_sqrt \
     --min-lr '1e-09' --warmup-updates 4000 --warmup-init-lr '1e-07' --weight-decay 0.01 --dropout 0.3 \
     --log-format 'simple' --log-interval 100 --max-tokens 8192 --save-interval-updates 2000 --max-update 200000 \
     --keep-interval-updates 10 --no-progress-bar --update-freq 8 --fp16 --seed 1  

# Pretrained Weights
We provide pre-trained weights of the models trained by AGG loss via Google Drive. You can reproduce the experimental performance of the paper by loading the corresponding weights.

 - [GPT-2 medium AGG model trained with WikiText-103 dataset](https://drive.google.com/file/d/18wz-87j0wpLQJP-WRLxUJNLnENNzk_km/view?usp=sharing)

 - [Transformer-base AGG model trained with WMT'14 English to German dataset](https://drive.google.com/file/d/12VHNuhWdHE4xYa2eBFqHYPzPVLS1V3jY/view?usp=sharing)

 - [Transformer-big AGG model trained with WMT'14 English to German dataset](https://drive.google.com/file/d/1J-HBMbQPDYliJtwCokFAU-4psfy5ABlg/view?usp=sharing)

# Evaluation
The commands below assume you are in the ```$AGG``` directory.

  ### Language Modeling
   - Evaluate PPL score for each token groups (high, medium, rare):
   
```
python -u eval_lm.py data-bin/wikitext-103_bpe --path checkpoints/$YOUR_CHECKPOINT_DIR/checkpoint_best.pt \
--batch-size 2 --tokens-per-sample 512 --context-window 400
```  

   - Evaluate Uniq score for each token groups (high, medium, rare):

```
python -u fairseq/custom/evaluation.py --batch-size-single-prediction 512 --batch-size-completion 48 \
--save-path ./eval_lm_metrics --ckpt best --model-path ./checkpoints/$YOUR_CHECKPOINT_DIR \
--data-dir ./data-bin/wikitext-103_bpe --base-dir ./ --eval-mode singletoken --data-split test
```
  
  ### Word Similarity
  
  ```
  python -u eval_emb.py --path-dir checkpoints/$YOUR_CHECKPOINT_DIR --path-file checkpoint_best.pt
  ```
  
  ### Neural Machine Translation
  
  - You can make average checkpoints through below script:
  
  ```
  python -u translation_eval/average_checkpoints.py --inputs checkpoints/$YOUR_CHECKPOINT_DIR --num-epoch-checkpoints 5 \
  --checkpoint-upper-bound 96 --output checkpoints/$YOUR_CHECKPOINT_DIR/checkpoint_avg.pt
  ```

  - Generate target sequences given test dataset from checkpoints:

  ```
  python -u data-bin/wmt14_en_de_joined --gen-subset test --path checkpoints/$YOUR_CHECKPOINT_DIR/checkpoint_avg.pt \
  --beam 4 --batch-size 100 --lenpen 0.6 --remove-bpe | tee translate.out
  ```
  
  - Since [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) used compound splitting, we also use compound splitting to calculate final BLEU score of the model trained with WMT'14 English to German dataset:
  
  ```
  bash translation_eval/compound_split_bleu.sh translate.out
  ```
  
# Reference

- Fairseq \
  https://github.com/pytorch/fairseq

- Neural Text Generation with Unlikelihood Training \
  https://github.com/facebookresearch/unlikelihood_training
      
