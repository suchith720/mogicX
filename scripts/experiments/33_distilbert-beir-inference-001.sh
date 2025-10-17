#!/bin/bash

datasets="arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

datasets="climate-fever"

output_file=outputs/33_distilbert-beir-inference-001.txt
for dataset in $datasets
do
	echo $dataset

	echo $dataset : >> $output_file

	suffix=$(echo $dataset | sed 's/\//-/g')
	if [ $dataset == "msmarco" ]
	then
		CUDA_VISIBLE_DEVICES=2,3 python mogicX/33_distilbert-beir-inference-001.py --dataset $dataset --prediction_suffix $suffix --save_test_prediction >> $output_file
	else
		CUDA_VISIBLE_DEVICES=2,3 python mogicX/33_distilbert-beir-inference-001.py --dataset $dataset --prediction_suffix $suffix --save_test_prediction --only_test >> $output_file
	fi
done

