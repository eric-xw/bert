# !/bin/bash
input_dir="/data1/xwang/summarization/bert/bert_input"
out_dir="/mnt/kobe/bert_features"

BERT_BASE_DIR=models/uncased_L-12_H-768_A-12

for filename in $(cat $input_dir/article_files.txt); do
#for filename in $input_dir/article/*; do
    echo $(basename $filename)
    CUDA_VISIBLE_DEVICES=0 python extract_features.py \
        --input_file=$input_dir/article/$filename \
        --output_file=$out_dir/article/$filename.jsonl \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --layers=-1 \
        --max_seq_length=512 \
        --batch_size=6
done