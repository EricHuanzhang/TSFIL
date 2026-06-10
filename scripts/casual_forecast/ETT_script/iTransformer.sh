export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name causal_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT/ \
  --data_path ETTh1.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data ETTh1c \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3 \
  --batch_size 32 \