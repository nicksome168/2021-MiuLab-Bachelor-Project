# MiuLab Bachelor Project

-   Task: [AICUP 2021 教育部全國大專校院人工智慧競賽 春季賽：醫病決策預判與問答](https://aidea-web.tw/topic/3665319f-cd5d-4f92-8902-00ebbd8e871d)
-   Results: [on wandb](https://wandb.ai/nicksome_yc/2021-Miu-Lab/workspace?workspace=user-nicksome_yc)

## Installation

-   Install packages in `Pipfile`:

    ```
    pipenv install
    ```

    or

    ```
    pipenv install --dev
    ```

    if you need to log training metrics by `wandb`.

    -   After `pipenv install`, run `pipenv shell` to get into the virtual environment you built.
    -   You can also manually install packages in `Pipfile` by `pip` or `conda` with Python 3.8

## Testing

-   Download trained model:
    -   Direct download:
        -   `qa_model.zip`: TBA
        -   Unzip and place models under `ckpt/`.
-   Test data is recommended to be placed at `data/qa/test.json`
-   The data of QA have to be preprocessed before predicting:
    ```
    python query_qa.py \
        --data_path data/qa/test.json \
        --model_name model_test.pkl \
        --processed_data_path data/qa/processed_test.json
    ```
-   Predict
    ```
    bash run.sh <path to context.json> <path to data.json> <path to save prediction.json>
    ```

## Training

-   Training data is strongly recommended to be placed at `data/qa/train.json`.
-   The data have to be preprocessed before training:
    ```
    python query_qa.py \
        --data_path data/qa/train.json \
        --model_name model_train.pkl \
        --processed_data_path data/qa/processed_train.json
    ```
-   We also use C3 dialog data to boost performance.
    -   Download `c3-d-train.json`, `c3-d-dev.json`, and `c3-d-test.json` at https://github.com/nlpdata/c3/tree/master/data and place them at `data/c3/train.json`, `data/c3/dev.json`, and `data/c3/test.json`.
-   To train:

    if you want to add DUMA layer, add `--add_duma` in the commandline

    -   only AICUP data
        ```
        python train_qa.py
        ```
    -   AICUP + c3
        ```
        python train_qa_c3.py
        ```
    -   AICUP + DREAM
        ```
        python train_qa_dream.py
        ```
    -   AICUP + c3 + DREAM

        ```
        python train_qa_all.py
        ```

    -   Validation data will be split from training data with a ratio of 10% automatically.
        -   The splitting process requires `data/rc/train.csv` to get the exact split as risk classification, so make sure the file exists.
    -   The program saves the best model by the accuracy of validation data (can be changed by `--metric_for_best`).
    -   Model will be saved in `--ckpt_dir` (default: `ckpt/qa`).
    -   Training uses `cuda:0` by default (can be changed by `--device`), and note that using `cpu` is not tested.
