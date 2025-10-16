#!/bin/bash

for i in $(seq 0 13)
do
	CUDA_VISIBLE_DEVICES=$i python mogicX/54_nvembed-for-msmarco-label-embeddings-002.py --idx $i --parts 14 &
done
wait
