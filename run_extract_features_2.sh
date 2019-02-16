# !/bin/bash
input_dir="/data1/xwang/summarization/bert/bert_input"
out_dir="/data1/xwang/summarization/bert/bert_output"

BERT_BASE_DIR=models/uncased_L-12_H-768_A-12

#for filename in $input_dir/highlight/*; do
#	echo $(basename $filename)
#	python extract_features.py \
#        --input_file=$filename \
#        --output_file=$out_dir/highlight/$(basename $filename).jsonl \
#        --vocab_file=$BERT_BASE_DIR/vocab.txt \
#        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#        --layers=-1 \
#        --max_seq_length=128 \
#        --batch_size=32
#done

for filename in $input_dir/article/*; do
	CUDA_VISIBLE_DEVICES=2 python extract_features.py \
        --input_file=$filename \
        --output_file=$out_dir/article/$(basename $filename).jsonl \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --layers=-1 \
        --max_seq_length=420 \
        --batch_size=6
done
