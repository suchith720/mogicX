#!/bin/bash

datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"

output_file=outputs/37_training-msmarco-distilbert-from-scratch-010.txt
for dataset in $datasets
do
	echo $dataset
	echo $dataset : >> $output_file
	python mogicX/37_training-msmarco-distilbert-from-scratch-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --dataset $dataset >> $output_file
done

