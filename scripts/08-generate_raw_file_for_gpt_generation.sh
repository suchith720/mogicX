datasets="nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"
for dataset in $datasets
do
	echo $dataset
	python scripts/07-generate_raw_file_for_gpt_generation.py --dataset $dataset
done


