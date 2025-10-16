#!/bin/bash

datasets="msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002_conflated.txt
for dataset in $datasets
do
	echo $dataset

	echo $dataset : >> $output_file
	suffix=$(echo $dataset | sed 's/\//-/g')

	if [ $dataset == "msmarco" ]
	then
		CUDA_VISIBLE_DEVICES=0,1 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --dataset $dataset --expt_no 0 --save_test_prediction --prediction_suffix $suffix >> $output_file
	else
		CUDA_VISIBLE_DEVICES=0,1 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --dataset $dataset --expt_no 0 --save_test_prediction --prediction_suffix $suffix --only_test >> $output_file
	fi

done

