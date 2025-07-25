export CUDA_VISIBLE_DEVICES=0,1

model_name=Autoformer_ALW
seq_len=512
red_len=256

for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs 1 \
  --red_len $red_len \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.01 \
  --itr 1

done
