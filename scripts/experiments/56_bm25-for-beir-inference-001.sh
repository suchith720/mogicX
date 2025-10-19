#!/bin/bash

datasets="msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

output_file=outputs/56_bm25-for-beir-inference-001.txt
for dataset in $datasets
do
	echo $dataset
	python mogicX/56_bm25-for-beir-inference-001.py --dataset $dataset >> $output_file
done

