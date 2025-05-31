# A Generative Outfit Recommendation System based on Occasion and Temperature
- Introduction to Artificial Intelligence @ NYCU 2025 Spring 


## Project layout
- `labels/`
    - `dataset.csv`
- `dataset/`
    - Unzip all images into this folder
- `caption/`
    - All tool that used to generate caption for images stored here

## Usage

- `embedding_to_prompt`
```shell
python .\train_embedding_to_prompt.py \
    --emb-path .\artifacts\final_emb_gender_onehot.npy \
    --prompt-path artifacts/image_prompts_with_static.txt \
    --output-dir checkpoints/embed2prompt \
    --base-model gpt2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --prefix-len 20 
```


- `embedding_predictor.py`
- Training
```shell
python .\embedding_predictor.py fit --csv .\labels\dataset.csv --out-dir .\artifacts\
```
- Predict
```shell
python embedding_predictor.py predict \
    --model artifacts/emb_mlp.pt \
    --gender 0 --temperature 2 --dresscode smart_casual
```