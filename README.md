# A Generative Outfit Recommendation System based on Occasion and Temperature
- Introduction to Artificial Intelligence @ NYCU 2025 Spring 
- Dataset is subset of [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal/tree/main)

## Project layout
- `labels/`
    - `dataset.csv`
- `dataset/`
    - Unzip all images into this folder
- `caption/`
    - All tool that used to generate caption for images stored here

## Usage
- Download `dataset.csv`, and place under `labels/`
- Download `dataset.zip` from [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal/tree/main), extract all images into `dataset/`
    - Only images present in `dataset.csv` will be used.
- Generate caption first using `python .\caption\generate_captions.py `, this will generate caption

- `train_embedding_to_prompt.py`
    - Training a prefix bridge model for generate imagegen prompt
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
- Training StyleEmbedding Predictor
```shell
python .\embedding_predictor.py fit --csv .\labels\dataset.csv --out-dir .\artifacts\
```
- Predict StyleEmbedding
```shell
python embedding_predictor.py predict \
    --model artifacts/emb_mlp.pt \
    --gender 0 --temperature 2 --dresscode smart_casual
```

- `generate_outfit.py`
```shell
python .\generate_outfit.py --temperature 0 --gender 0 --dresscode 4 --out result-0-0-4_4.png
```