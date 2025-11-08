#!/bin/bash

if [ $# -lt 1 ]
then
	echo "bash scripts/experiments/30_ngame-for-msmarco-with-hard-negatives-mteb-inference.sh <expt_no>" 
	exit 1
fi

datasets="msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

datasets="arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

# output_file=outputs/30_ngame-for-msmarco-with-hard-negatives-$(printf "%03d" $1).txt

output_file=outputs/30_ngame-for-msmarco-with-hard-negatives-pretrained.txt
for dataset in $datasets
do
	echo $dataset

	echo $dataset : >> $output_file
	CUDA_VISIBLE_DEVICES=0,1,2,3 python mogicX/30_ngame-for-msmarco-with-hard-negatives-mteb-inference.py --dataset $dataset --expt_no $1 \
		--use_pretrained >> $output_file
done

