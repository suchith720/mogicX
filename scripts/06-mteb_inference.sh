#!/bin/bash

# datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"

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

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --dataset $dataset >> $output_file
# done

# output_file=outputs/44_distilbert-category-oracle-for-msmarco-004.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/44_distilbert-category-oracle-for-msmarco-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --dataset $dataset >> $output_file
# done

# output_file=outputs/48_oak-distilbert-for-msmarco-from-scratch-with-category-metadata-001.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/48_oak-distilbert-for-msmarco-from-scratch-with-category-metadata-mteb-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --dataset $dataset >> $output_file
# done

# output_file=outputs/msmarco-distilbert-dot-v5_category-gpt-linker.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/44_distilbert-category-oracle-for-msmarco-mteb-inference.py --dataset $dataset --use_pretrained >> $output_file
# done

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002_conflated.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --only_test --do_test_inference --dataset $dataset --expt_no 2 >> $output_file
# done

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-002_conflated-001.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	CUDA_VISIBLE_DEVICES=0,1 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --only_test --do_test_inference --dataset $dataset --expt_no 3 >> $output_file
# done

# output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-003.txt
# for dataset in $datasets
# do
# 	echo $dataset
# 	echo $dataset : >> $output_file
# 	CUDA_VISIBLE_DEVICES=0,1 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --only_test --do_test_inference --dataset $dataset --expt_no 2 >> $output_file
# done

# output_file=outputs/50_bert-ngame-category-linker-oracle-for-msmarco-005.txt
# for dataset in $datasets
# do
# 	echo $dataset : 
# 	CUDA_VISIBLE_DEVICES=0,1 python mogicX/50_bert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --dataset $dataset --expt_no 5 >> $output_file
# done

if [ $# -lt 1 ]
then
	echo "bash scripts/06-mteb_inference.sh <expt_no>" 
	exit 1
fi

datasets="arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid cqadupstack/android \
        cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats \
        cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

output_file=outputs/50_distilbert-ngame-category-linker-oracle-for-msmarco-$(printf "%03d" $1).txt
for dataset in $datasets
do
	echo $dataset : >> $output_file
	CUDA_VISIBLE_DEVICES=0,1 python mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-mteb-inference.py --dataset $dataset --expt_no $1 >> $output_file
done

