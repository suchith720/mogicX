expt_no="1 3 4 5 6 7 8 9"

for n in $expt_no 
do
	echo Experiment $n: 
	python mogicX/37_training-msmarco-distilbert-from-scratch-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --expt_no $n >> outputs/37_training-msmarco-distilbert-from-scratch.txt
done
