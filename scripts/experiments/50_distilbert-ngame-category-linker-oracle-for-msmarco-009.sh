#!/bin/bash

datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"

output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-009.txt
for dataset in $datasets
do
	echo $dataset : 
	CUDA_VISIBLE_DEVICES=4,5 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --dataset $dataset --expt_no 9 >> $output_file
done

