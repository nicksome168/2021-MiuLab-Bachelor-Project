# HW2 Chinese Question Answering
- [Description Slide](https://docs.google.com/presentation/d/1eonDCBNEqbvAEGKqPWt3Ew1JjVlBYXX45G2Hqs7c0Hk)
- No Kaggle competition

## Installation
- Install packages in `Pipfile`.
- Put data (csv files) in `data/`.

## Training
- Choose the pretrained model to use in `model.py` and tokenizer in `train_*.py`
- Context selection
```
python3 train_context_selection.py
```
- Question answering
```
python3 train_qa.py
```
- The program saves the best model by the exact match of validation data (can be changed in args).
- Model will be saved in `ckpt/`.

## Prediction
- Download trained model
```
bash download.sh
```
- Predict
```
bash run.sh <path to context.json> <path to data.json> <path to save prediction.json>
```
