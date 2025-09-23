#!/bin/bash

datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002_thresh-02.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --dataset $dataset --build_block >> $output_file
# done

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-004.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	CUDA_VISIBLE_DEVICES=2,3 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --only_test --do_test_inference --dataset $dataset --expt_no 4 >> $output_file
# done

output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002-combined.txt
for dataset in $datasets
do
	echo $dataset
	echo $dataset : >> $output_file
	CUDA_VISIBLE_DEVICES=0,1 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --only_test --do_test_inference --dataset $dataset --expt_no 5 >> $output_file
done
