#!/bin/bash

datasets="msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

TYPE=001

if [ $TYPE == "001" ]
then
	if [ $# -lt 1 ]
	then
		echo bash scripts/04-create_beir_early-fusion_setup.sh "<expt_no>"
		exit 1
	fi

	for dataset in $datasets
	do
		echo $dataset

		# python scripts/02-get_raw_and_config_for_oracle.py --dataset $dataset --expt_no $1 --task raw 
		# python scripts/02-get_raw_and_config_for_oracle.py --dataset $dataset --expt_no $1 --task config

		python scripts/02-get_raw_and_config_for_oracle.py --dataset $dataset --expt_no $1 --task raw --type prediction \
			--save_dir_name wiki_entities
		python scripts/02-get_raw_and_config_for_oracle.py --dataset $dataset --expt_no $1 --task config --type prediction \
			--save_dir_name wiki_entities
	done

elif [ $TYPE == "002" ]
then
	cnt=0
	for dataset in $datasets
	do
		echo $dataset : 
		ls /data/datasets/$dataset/qrels/ 
		n=$(ls /data/datasets/$dataset/qrels/ | grep train | wc -l) 
		if [ $n -gt 0 ]
		then
			((cnt++))
		fi
	done
	echo $cnt

elif [ $TYPE == "003" ]
then
	for dataset in $datasets
	do
		n_lbl=$(wc -l /data/datasets/$dataset/XC/raw_data/label.raw.csv | awk '{print $1}')
		n_data=$(wc -l /data/datasets/$dataset/XC/raw_data/test.raw.csv | awk '{print $1}')
		echo \'$dataset\' : { \'n_lbl\':$((n_lbl-1)), \'n_data\':$((n_data)) },
	done
else
	echo Invalid TYPE: $TYPE
fi

