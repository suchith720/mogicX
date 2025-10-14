#!/bin/bash

# datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"
# for dataset in $datasets
# do
# 	echo $dataset : 
# 	python  mogicX/01-msmarco-gpt-entity-linker-for-mteb-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --save_test_prediction --dataset $dataset --prediction_suffix $dataset
# 	cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/test_predictions_$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/entity-gpt_ngame_tst_X_Y.npz
# 	cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/label_predictions_$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/entity-gpt_ngame_lbl_X_Y.npz
# 	cp /data/datasets/msmarco/XC/raw_data/entity_gpt.raw.txt /home/aiscuser/scratch1/datasets/$dataset/XC/raw_data/entity-gpt_ngame.raw.txt
# done

# datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"
# for dataset in $datasets
# do
# 	echo $dataset : 
# 	python  mogicX/01-msmarco-gpt-wikipedia-entity-linker-for-mteb-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --save_test_prediction --dataset $dataset --prediction_suffix wiki-$dataset
# 	cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/test_predictions_wiki-$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/wiki-entity_ngame_tst_X_Y.npz
# 	cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/label_predictions_$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/wiki-entity_ngame_lbl_X_Y.npz
# 	cp /data/from_b/wiki_entity.raw.csv /home/aiscuser/scratch1/datasets/$dataset/XC/raw_data/wiki-entity_ngame.raw.csv
# done

# datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"
# datasets="msmarco"
# for dataset in $datasets
# do
# 	echo $dataset : 
# 	python  mogicX/01-msmarco-gpt-wikipedia-entity-linker-for-mteb-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --save_representation --dataset $dataset --prediction_suffix wiki-$dataset 
# 
# 	# python  mogicX/01-msmarco-gpt-wikipedia-entity-linker-for-mteb-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --save_test_prediction --dataset $dataset --prediction_suffix wiki-$dataset 
# 
# 	# python  mogicX/01-msmarco-gpt-wikipedia-entity-linker-for-mteb-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_train_inference --save_train_prediction --do_test_inference --save_test_prediction --dataset $dataset --prediction_suffix wiki-$dataset
# 
# 	# python  mogicX/01-msmarco-gpt-wikipedia-entity-linker-for-mteb-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_train_inference --save_train_prediction --dataset $dataset --prediction_suffix wiki-$dataset
# 
# 	# cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/test_predictions_wiki-$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/wiki-entity_ngame_tst_X_Y.npz
# 
# 	# cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/train_predictions_wiki-$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/wiki-entity_ngame_trn_X_Y.npz
# 
# 	# cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/label_predictions_wiki-$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/wiki-entity_ngame_lbl_X_Y.npz
# 
# 	# cp /data/from_b/wiki_entity.raw.csv /home/aiscuser/scratch1/datasets/$dataset/XC/raw_data/wiki-entity_ngame.raw.csv
# done

# datasets="nq" 
# for dataset in $datasets
# do
# 	echo $dataset : 
# 	python  mogicX/01-msmarco-gpt-entity-linker-for-mteb-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_train_inference --save_train_prediction --dataset $dataset --prediction_suffix $dataset
# 	cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/train_predictions_$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/entity-gpt_ngame_trn_X_Y_kaggle.npz
# 	cp /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/train-label_predictions_$dataset.npz /home/aiscuser/scratch1/datasets/$dataset/XC/entity-gpt_ngame_trn-lbl_X_Y_kaggle.npz
# done

# datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"
# for dataset in $datasets
# do
# 	echo $dataset : 
# 	python scripts/13-msmarco_gpt_category_linker_mteb_specific_category_inference.py --expt_no 1 --dataset $dataset
# done

if [ $# -lt 1 ]
then
	echo "bash scripts/05-mteb_linker.sh <expt_no>" 
	exit 1
fi

datasets="arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid cqadupstack/android \
        cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats \
        cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

for dataset in $datasets
do
	echo $dataset
	suffix=$(echo $dataset | sed 's/\//-/g')
	CUDA_VISIBLE_DEVICES=0,1 python mogicX/47_msmarco-gpt-category-linker-mteb-inference.py --dataset $dataset --prediction_suffix $suffix --expt_no $1
done

