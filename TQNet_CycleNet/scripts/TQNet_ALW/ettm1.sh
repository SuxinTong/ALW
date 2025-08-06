model_name=TQNet_ALW

root_path_name=./dataset/ETT-small/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

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
  --enc_in 7 \
  --cycle 96 \
  --dropout 0.5 \
  --train_epochs 5 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.0005

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
  --enc_in 7 \
  --cycle 96 \
  --dropout 0.5 \
  --train_epochs 3 \
  --patience 5 \
  --itr 1 --batch_size 64 --learning_rate 0.001

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
  --enc_in 7 \
  --cycle 96 \
  --dropout 0.5 \
  --train_epochs 3 \
  --patience 5 \
  --itr 1 --batch_size 32 --learning_rate 0.001

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
  --enc_in 7 \
  --cycle 96 \
  --dropout 0.5 \
  --train_epochs 10 \
  --patience 5 \
  --itr 1 --batch_size 64 --learning_rate 0.003


# for train_epochs in 1 3 5 10
# do

# for batch_size in 16 32 64 128 256
# do

# for learning_rate in 0.005 0.003 0.001 0.0005 0.0003
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
#       --enc_in 7 \
#       --cycle 96 \
#       --red_len $red_len \
#       --train_epochs $train_epochs \
#       --patience 5 \
#       --dropout 0.5 \
#       --itr 1 --batch_size $batch_size --learning_rate $learning_rate
# done

# done

# done

# done
