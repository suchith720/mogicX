datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"
# for dataset in $datasets
# do
# 	echo $dataset : 
# 	python scripts/get_data_ngame-category-linker_for_oracle.py --dataset $dataset
# done
# 
# cnt=0
# for dataset in $datasets
# do
# 	echo $dataset : 
# 	ls /data/datasets/$dataset/qrels/ 
# 	n=$(ls /data/datasets/$dataset/qrels/ | grep train | wc -l) 
# 	if [ $n -gt 0 ]
# 	then
# 		((cnt++))
# 	fi
# done
# echo $cnt

for dataset in $datasets
do
	n_lbl=$(wc -l /data/datasets/$dataset/XC/raw_data/label.raw.csv | awk '{print $1}')
	n_data=$(wc -l /data/datasets/$dataset/XC/raw_data/test.raw.csv | awk '{print $1}')
	echo \'$dataset\' : { \'n_lbl\':$((n_lbl-1)), \'n_data\':$((n_data)) },
done

