#!/bin/bash

datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"

# output_file=outputs/33_ngame-mteb-inference-001.txt
# for dataset in $datasets
# do
# 	echo $dataset :
# 	echo $dataset : >> $output_file
# 	python mogicX/33_ngame-mteb-inference-001.py  --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --dataset $dataset --build_block >> $output_file
# done

# output_file=outputs/33_oak-mteb-inference-002.txt
# for dataset in $datasets
# do
# 	echo $dataset :
# 	echo $dataset : >> $output_file
# 	python mogicX/33_oak-mteb-inference-002.py  --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --dataset $dataset --build_block >> $output_file
# done

# output_file=outputs/33_distilbert-mteb-inference-003.txt
# for dataset in $datasets
# do
# 	echo $dataset :
# 	echo $dataset : >> $output_file
# 	python mogicX/33_ngame-mteb-inference-001.py  --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --use_pretrained --dataset $dataset >> $output_file
# done

# output_file=outputs/33_distilbert-dot-mteb-inference-004.txt
# for dataset in $datasets
# do
# 	echo $dataset :
# 	echo $dataset : >> $output_file
# 	python mogicX/33_ngame-dot-mteb-inference-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --use_pretrained --dataset $dataset >> $output_file
# done

output_file=outputs/33_oak-with-wiki-entity-mteb-zs-inference-005.txt
for dataset in $datasets
do
	echo $dataset : 
	echo $dataset : >> $output_file
	python mogicX/33_oak-with-wiki-entity-mteb-zs-inference-003.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --dataset $dataset >> $output_file
done
