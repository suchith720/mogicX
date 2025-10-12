#!/bin/bash

datasets="msmarco arguana climate-fever dbpedia fever fiqa hotpotqa nfcorpus nq quora scidocs scifact touche2020 trec-covid cqadupstack/android \
	cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats \
	cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

# for dset in $datasets
# do
# 	mkdir -p beir/$dset
# 	mv beir/$dset'_'* beir/$dset/
# 	mv beir/$dset'-'* beir/$dset/
# done

for dset in $datasets
do
	rm -rf beir/$dset/$dset'_data-category-gpt-linker_conflated-001_conflated-001.json'
	rm -rf beir/$dset/$dset'_data-gpt-category-linker-ngame-linker_conflated-001_conflated-001.json'
	rm -rf beir/$dset/$dset'_data-gpt-category-linker-ngame-linker_conflated-001_conflated-001_008.json'
	rm -rf beir/$dset/$dset'_data-gpt-category-linker-ngame-linker_conflated-001_conflated-001_009.json'
done
