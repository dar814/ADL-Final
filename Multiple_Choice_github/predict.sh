python predict.py\
  --test_dataset $1\
  --config_path model/${2^}/config.json\
  --tokenizer_path model/${2^}/\
  --model_path model/${2^}/\
  --dataset_type $2\
  --max_seq_length 512\
  --per_device_test_batch_size 4 \
  --output_predict_path $3 \