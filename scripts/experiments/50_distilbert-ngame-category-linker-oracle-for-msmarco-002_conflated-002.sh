#!/bin/bash

datasets="msmarco nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"

output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002_conflated-002.txt
for dataset in $datasets
do
	echo $dataset
	echo $dataset : >> $output_file
	CUDA_VISIBLE_DEVICES=2,3 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --only_test --do_test_inference --dataset $dataset --expt_no 4 >> $output_file
done
