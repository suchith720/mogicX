#!/bin/bash

datasets="msmarco nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"

TYPE=001

if [ $TYPE == "001" ]
then
	if [ $# -lt 1 ]
	then
		echo "<expt_no>"
		exit 1
	fi

	for dataset in $datasets
	do
		echo $dataset : 
		python scripts/02-get_data_ngame-category-linker_for_oracle.py --dataset $dataset --expt_no $1 --type raw
		python scripts/02-get_data_ngame-category-linker_for_oracle.py --dataset $dataset --expt_no $1 --type config
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

