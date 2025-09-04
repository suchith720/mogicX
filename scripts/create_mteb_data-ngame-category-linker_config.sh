datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"
for dataset in $datasets
do
	echo $dataset : 
	python scripts/get_data_ngame-category-linker_for_oracle.py --dataset $dataset
done
