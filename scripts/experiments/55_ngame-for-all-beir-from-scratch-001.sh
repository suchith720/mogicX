#!/bin/bash

datasets="msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

datasets="scidocs"

output_file=outputs/55_ngame-for-all-beir-from-scratch-001.txt

for dataset in $datasets
do
	echo $dataset

	echo $dataset : >> $output_file
	suffix=$(echo $dataset | sed 's/\//-/g')

	CUDA_VISIBLE_DEVICES=2,3 python mogicX/55_ngame-for-all-beir-from-scratch-beir-inference.py --dataset $dataset >> $output_file

done



