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
# 	echo $dataset : >> $output_file
# 	python mogicX/33_ngame-mteb-inference-001.py  --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --use_pretrained --dataset $dataset >> $output_file
# done

# output_file=outputs/33_distilbert-dot-mteb-inference-004.txt
# for dataset in $datasets
# do
# 	echo $dataset : >> $output_file
# 	python mogicX/33_ngame-dot-mteb-inference-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --use_pretrained --dataset $dataset >> $output_file
# done

# output_file=outputs/33_oak-wiki-entity-mteb-inference-005.txt
# for dataset in $datasets
# do
# 	echo $dataset : >> $output_file
# 	python mogicX/33_oak-with-wiki-entity-mteb-zs-inference-003.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --dataset $dataset >> $output_file
# done

# output_file=outputs/35_metadexa-mteb-inference-001.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/35_metadexa-mteb-inference-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --dataset $dataset --only_test >> $output_file
# done

# output_file=outputs/30_ngame-for-msmarco-with-hard-negatives-003.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/30_ngame-for-msmarco-with-hard-negatives-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --dataset $dataset --only_test >> $output_file
# done

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --dataset $dataset >> $output_file
# done

output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002.txt
for dataset in $datasets
do
	echo $dataset
	echo $dataset : >> $output_file
	python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --dataset $dataset >> $output_file
done

