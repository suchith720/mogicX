#!/bin/bash

datasets="arguana msmarco climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

datasets="arguana"

output_file=outputs/37_training-msmarco-distilbert-from-scratch-008-verify.txt
for dataset in $datasets
do
	echo $dataset

	# echo $dataset : >> $output_file
	# CUDA_VISIBLE_DEVICES=3 python mogicX/37_training-msmarco-distilbert-from-scratch-mteb-inference-001.py --dataset $dataset >> $output_file
	
	CUDA_VISIBLE_DEVICES=3 python mogicX/37_training-msmarco-distilbert-from-scratch-mteb-inference-001.py --dataset $dataset
done
