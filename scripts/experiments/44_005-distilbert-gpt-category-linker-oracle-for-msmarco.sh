#!/bin/bash

# datasets="arguana webis-touche2020 cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics \
# 	cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress msmarco climate-fever \

# datasets="arguana webis-touche2020 dbpedia-entity nfcorpus trec-covid"

datasets="arguana msmarco climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

output_file=outputs/44_distilbert-gpt-category-linker-oracle-for-msmarco-005.txt
for dataset in $datasets
do
	echo $dataset
	echo $dataset : >> $output_file
	python mogicX/44_distilbert-gpt-category-linker-oracle-for-msmarco-beir-inference.py --dataset $dataset >> $output_file
done
