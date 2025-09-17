#!/bin/bash

output_dir=/data/outputs/mogicX/47_msmarco-gpt-category-linker-001/

trn_file="/data/datasets/msmarco/XC/category-gpt_trn_X_Y.npz"
tst_file="/data/datasets/msmarco/XC/category-gpt_tst_X_Y.npz"
lbl_file="/data/datasets/msmarco/XC/raw_data/category-gpt.raw.csv"

# pred_file=$output_dir/predictions/test_predictions.npz
pred_file=$output_dir/predictions/train_predictions.npz
embed_file=$output_dir/predictions/label_repr.pth

TYPE="final"
NUMBER=002

if [ $TYPE == "experiment" ]
then
	output_dir=outputs/47_msmarco-gpt-category-linker-001
	mkdir -p $output_dir
	n=$(ls $output_dir | wc -l); ((n++)); n=$(printf "%03d" $n)
	output_dir=$output_dir/conflation_$n

	echo $output_dir
	
	python mogicX/42_entity-conflation.py --pred_file $pred_file \
		--lbl_info_file $lbl_file \
		--trn_file $trn_file \
		--tst_file $tst_file \
		--embed_file $embed_file \
		--output_dir $output_dir \
		--topk 5 \
		--pred_lbl_freq 100 \
		--batch_size 1024 \
		--freq_thresh 50 \
		--sim_score_thresh 50 \
		--max_size_thresh 100 \
		--conflated_lbl_freq 500 \
		--lbl_cluster_sz 2 \
		--type concat \
		--print_stats \
		--output_prefix 001 \
		--exclude_invalid
	
	if [ -d $output_dir ]
	then
		lbl_file=$output_dir/raw_data/category-gpt_conflated-001.raw.csv
		grep ' || ' $lbl_file >> $output_dir/conflations.txt
	fi

elif [ $TYPE == "final" ]
then
	if [ $NUMBER == "001" ]
	then
		python mogicX/42_entity-conflation.py --pred_file $pred_file \
			--lbl_info_file $lbl_file \
			--trn_file $trn_file \
			--tst_file $tst_file \
			--embed_file $embed_file \
			--topk 5 \
			--pred_lbl_freq 100 \
			--batch_size 1024 \
			--freq_thresh 50 \
			--sim_score_thresh 50 \
			--max_size_thresh 100 \
			--conflated_lbl_freq 500 \
			--lbl_cluster_sz 2 \
			--type mid \
			--print_stats \
			--output_prefix 001 \
			--exclude_invalid

	elif [ $NUMBER == "002" ]
	then
		python mogicX/42_entity-conflation.py --pred_file $pred_file \
			--lbl_info_file $lbl_file \
			--trn_file $trn_file \
			--tst_file $tst_file \
			--embed_file $embed_file \
			--topk 5 \
			--pred_lbl_freq 100 \
			--batch_size 1024 \
			--freq_thresh 50 \
			--sim_score_thresh 50 \
			--max_size_thresh 100 \
			--conflated_lbl_freq 500 \
			--type mid \
			--print_stats \
			--output_prefix 002 \
			--exclude_invalid
	else
		python mogicX/42_entity-conflation.py --pred_file $pred_file \
			--lbl_info_file $lbl_file \
			--trn_file $trn_file \
			--tst_file $tst_file \
			--embed_file $embed_file \
			--topk 3 \
			--pred_score_thresh 0.2 \
			--diff_thresh 0.1 \
			--batch_size 1024 \
			--freq_thresh 25 \
			--sim_score_thresh 25 \
			--min_size_thresh 2 \
			--max_size_thresh 100 \
			--type mid \
			--print_stats
	fi
fi

