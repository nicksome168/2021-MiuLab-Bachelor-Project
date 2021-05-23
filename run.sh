python3.8 predict.py \
    --context_path $1 \
    --data_path $2 \
    --cs_ckpt_path ckpt/cs_best.pt \
    --qa_ckpt_path ckpt/qa_best.pt \
    --pred_path $3 \
    --batch_size 12
