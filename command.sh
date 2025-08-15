# run inference on full msmarco dataset
python mogicX/00_ngame-for-msmarco-inference-001.py --use_pretrained --do_test_inference --save_test_prediction --use_sxc_sampler --only_test --pickle_dir /home/aiscuser/scratch1/datasets/processed/

# training linker with extracted entities from MSMARCO
python mogicX/01-msmarco-gpt-entity-linker-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --build_block --use_pretrained

python mogicX/01-msmarco-llama-entity-linker-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --build_block --use_pretrained

# train ngame on the new LF-WikiSeeAlsoTitles-320K dump
python mogicX/02_ngame-for-wikiseealsotitles-20250123-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --build_block --do_test_inference --use_pretrained

python mogicX/02_ngame-for-wikiseealsotitles-20250123-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --build_block --do_test_inference --use_pretrained

# infer on the old LF-WikiSeeAlsoTitles-320K to verify the pipeline
python mogicX/03_ngame-for-wikiseealsotitles-001.py  --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --build_block --do_test_inference --use_pretrained

# infer ngame on the new LF-WikiSeeAlsoTitles-320K dump
python mogicX/02_ngame-for-wikiseealsotitles-20250123-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --save_test_prediction

# mogic meta-encoder inference
python mogicX/05_mogic-meta-encoder-for-wikiseealsotitles-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --build_block --do_test_inference

# linker on wikiseealsotitles split
python mogicX/06_ngame-linker-for-wikiseealsotitles-split-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --build_block --do_test_inference

python mogicX/01-msmarco-gpt-entity-linker-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --do_train_inference --save_test_prediction --save_train_prediction --prediction_suffix full

# MSMARCO
python mogicX/00_ngame-for-msmarco-inference-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --save_test_prediction --only_test --use_pretrained

python mogicX/00_ngame-for-msmarco-inference-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --save_test_prediction --only_test --use_pretrained --prediction_suffix full

python mogicX/23_oak-for-msmarco-with-gpt-entity-linker-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --do_test_inference --use_pretrained --only_test

python mogicX/25_sbert-for-msmarco-inference-001.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test

python mogicX/37_training-msmarco-distilbert-from-scratch-002.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --exact --do_test_inference --use_pretrained

# MSMARCO distilbert inference
python mogicX/37_training-msmarco-distilbert-from-scratch-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference --expt_no 8

python mogicX/30_ngame-for-msmarco-with-hard-negatives-inference.py --use_sxc_sampler --pickle_dir /home/aiscuser/scratch1/datasets/processed/ --only_test --do_test_inference

# Conflation code
python mogicX/42_entity-conflation.py --pred_file /data/datasets/msmarco/XC/entity-gpt_ngame_trn_X_Y.npz --lbl_file //data/datasets/msmarco/XC/raw_data/entity_gpt.raw.txt  --trn_file /data/datasets/msmarco/XC/entity_gpt_trn_X_Y.npz --tst_file /data/datasets/msmarco/XC/entity_gpt_tst_X_Y.npz --topk 3 --batch_size 1024 --min_thresh 2 --max_thresh 100 --freq_thresh 50 --score_thresh 25 --diff_thresh 0.1 --embed_file /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/label_repr_full.pth
