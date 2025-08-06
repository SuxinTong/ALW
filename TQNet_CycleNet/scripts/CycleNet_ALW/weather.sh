model_name=CycleNet_ALW

root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

model_type='mlp'
seq_len=512
red_len=256

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'96 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --cycle 144 \
  --red_len $red_len \
  --model_type $model_type \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.05

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'192 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --cycle 144 \
  --red_len $red_len \
  --model_type $model_type \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.003

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'336 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --cycle 144 \
  --red_len $red_len \
  --model_type $model_type \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.05

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'720 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --cycle 144 \
  --red_len $red_len \
  --model_type $model_type \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.05
