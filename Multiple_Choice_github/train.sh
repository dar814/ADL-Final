python train.py\
  --model_name_or_path bert-base-cased\
  --train_file $1 \
  --validation_file $2\
  --dataset_type $4\
  --max_seq_length 128\
  --logging_strategy steps\
  --logging_steps 1\
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 64 \
  --learning_rate 2e-4 \
  --num_train_epochs 10\
  --overwrite_output_dir \
  --output_dir $3 \
  --do_train \
  --do_eval \
#   --max_train_samples 50\
#   --max_eval_samples 10