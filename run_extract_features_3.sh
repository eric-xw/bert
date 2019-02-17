# !/bin/bash
input_dir="/data1/xwang/summarization/bert/bert_input"
out_dir="/data1/xwang/summarization/bert/bert_output"

BERT_BASE_DIR=models/uncased_L-12_H-768_A-12

for filename in $(cat $input_dir/highlight_files.txt); do
    echo $(basename $filename)
    CUDA_VISIBLE_DEVICES=3 python extract_features.py \
        --input_file=$input_dir/highlight/$filename \
        --output_file=$out_dir/highlight/$filename.jsonl \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --layers=-1 \
        --max_seq_length=160 \
        --batch_size=32
done
