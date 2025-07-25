model_name=CycleNet_ALW

root_path_name=./dataset/exchange_rate/
data_path_name=exchange_rate.csv
model_id_name=exchange
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
  --enc_in 8 \
  --cycle 24 \
  --red_len $red_len \
  --model_type $model_type \
  --train_epochs 4 \
  --itr 1 --batch_size 256 --learning_rate 0.01

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
  --enc_in 8 \
  --cycle 24 \
  --red_len $red_len \
  --model_type $model_type \
  --train_epochs 2 \
  --itr 1 --batch_size 64 --learning_rate 0.01

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
  --enc_in 8 \
  --cycle 24 \
  --red_len $red_len \
  --model_type $model_type \
  --train_epochs 3 \
  --patience 5 \
  --itr 1 --batch_size 128 --learning_rate 0.01

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
  --enc_in 8 \
  --cycle 24 \
  --red_len $red_len \
  --model_type $model_type \
  --train_epochs 3 \
  --patience 5 \
  --itr 1 --batch_size 128 --learning_rate 0.01

# for train_epochs in 1 2 3 4 5
# do

# for batch_size in 16 32 64 128 256
# do

# for learning_rate in 0.05 0.01 0.005 0.001 0.0005 0.0002 0.0001
# do

# for pred_len in 96 192 336 720
# do
#     python -u run.py \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 8 \
#       --cycle 24 \
#       --red_len $red_len \
#       --model_type $model_type \
#       --train_epochs $train_epochs \
#       --patience 5 \
#       --itr 1 --batch_size $batch_size --learning_rate $learning_rate
# done

# done

# done

# done