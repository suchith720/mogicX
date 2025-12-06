#!/bin/bash

datasets="msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002_thresh-02.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --dataset $dataset --build_block >> $output_file
# done

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-004.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	CUDA_VISIBLE_DEVICES=2,3 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --only_test --do_test_inference --dataset $dataset --expt_no 4 >> $output_file
# done

datasets="cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-008-007.txt
for dataset in $datasets
do
	echo $dataset
	suffix="${dataset//\//-}"

	echo $dataset : >> $output_file
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --only_test \
		--save_representation --dataset $dataset --expt_no 15 --prediction_suffix $suffix-008-007 >> $output_file
done
