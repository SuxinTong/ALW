export CUDA_VISIBLE_DEVICES=0,1

model_name=iTransformer_ALW
seq_len=512
red_len=256

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --red_len $red_len \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.003 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --red_len $red_len \
  --d_model 64 \
  --d_ff 64 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
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
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs 1 \
  --red_len $red_len \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.005 \
  --itr 1


# for train_epochs in 1 10
# do

# for batch_size in 8 16 32 64 128
# do 

# for learning_rate in 0.0005 0.001 0.003 0.005 0.01
# do

# for e_layers in 1 2
# do

# for d_model in 64 128 256 512
# do

# for pred_len in 96 192 336 720
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --train_epochs $train_epochs \
#   --red_len $red_len \
#   --d_model $d_model \
#   --d_ff $d_model \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --itr 1

# done

# done

# done

# done

# done

# done