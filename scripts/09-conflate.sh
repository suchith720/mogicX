#!/bin/bash

output_dir=/data/outputs/mogicX/47_msmarco-gpt-category-linker-001/

trn_file="/data/datasets/msmarco/XC/category-gpt_trn_X_Y.npz"
tst_file="/data/datasets/msmarco/XC/category-gpt_tst_X_Y.npz"
lbl_file="/data/datasets/msmarco/XC/raw_data/category-gpt.raw.csv"

pred_file=$output_dir/predictions/test_predictions.npz
embed_file=$output_dir/predictions/label_repr.pth

python mogicX/42_entity-conflation.py --pred_file $pred_file --lbl_info_file $lbl_file --trn_file $trn_file \
	--tst_file $tst_file --topk 10 --pred_score_thresh 0.2 --diff_thresh 0.1 \
	--batch_size 1024 --freq_thresh 50 --sim_score_thresh 25 \
	--min_thresh 2 --max_thresh 100 --embed_file $embed_file 

