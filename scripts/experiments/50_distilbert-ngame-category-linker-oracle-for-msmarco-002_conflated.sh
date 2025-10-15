#!/bin/bash

if [ $# -lt 1 ]
then
	echo "bash scripts/06-beir_inference.sh <expt_no>" 
	exit 1
fi

datasets="fever fiqa nq cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica \
	cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix"

output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002_conflated.txt
for dataset in $datasets
do
	echo $dataset : >> $output_file
	CUDA_VISIBLE_DEVICES=4,5 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --dataset $dataset --expt_no $1 >> $output_file
done

