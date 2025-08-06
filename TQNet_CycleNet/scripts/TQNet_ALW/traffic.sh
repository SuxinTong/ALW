model_name=TQNet_ALW

root_path_name=./dataset/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

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
  --enc_in 862 \
  --cycle 168 \
  --train_epochs 30 \
  --use_pe 0 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.005

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
  --enc_in 862 \
  --cycle 168 \
  --train_epochs 30 \
  --use_pe 0 \
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
  --enc_in 862 \
  --cycle 168 \
  --train_epochs 30 \
  --use_pe 0 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.005

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
  --enc_in 862 \
  --cycle 168 \
  --train_epochs 30 \
  --use_pe 0 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.005



# for pred_len in 96 192 336 720
# do
# for learning_rate in 0.005 0.003 0.001
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
#       --enc_in 862 \
#       --cycle 168 \
#       --train_epochs 30 \
#       --use_pe 0 \
#       --patience 5 \
#       --itr 1 --batch_size 32 --learning_rate $learning_rate
# done
# done
