export CUDA_VISIBLE_DEVICES=0,1

model_name=iTransformer_ALW
seq_len=512
red_len=256

for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --red_len $red_len \
  --d_model 512 \
  --d_ff 512 \
  --use_pe 0 \
  --learning_rate 0.001 \
  --itr 1

done


