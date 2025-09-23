#!/bin/bash

for i in $(seq 0 5)
do
	CUDA_VISIBLE_DEVICES=$i python mogicX/51_finetune-llama-for-category-generation.py --index $i --parts 6 --type infer --bit 8 &
done

wait
