# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/37_training-msmarco-distilbert-from-scratch.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/37_training-msmarco-distilbert-from-scratch.ipynb 2
import os
# os.environ['HIP_VISIBLE_DEVICES'] = '6,7,8,9'
os.environ['HIP_VISIBLE_DEVICES'] = '0,1,8,9'

import torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse

from transformers import DistilBertConfig

from xcai.basics import *
from xcai.models.PPP0XX import DBT023

# %% ../nbs/37_training-msmarco-distilbert-from-scratch.ipynb 4
os.environ['WANDB_PROJECT'] = 'mogicX_00-msmarco'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_block', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')
    
    parser.add_argument('--do_train_inference', action='store_true')
    parser.add_argument('--do_test_inference', action='store_true')
    
    parser.add_argument('--save_train_prediction', action='store_true')
    parser.add_argument('--save_test_prediction', action='store_true')
    parser.add_argument('--save_label_prediction', action='store_true')
    
    parser.add_argument('--save_representation', action='store_true')
    
    parser.add_argument('--use_sxc_sampler', action='store_true')
    parser.add_argument('--only_test', action='store_true')

    parser.add_argument('--pickle_dir', type=str, required=True)
    
    parser.add_argument('--prediction_suffix', type=str, default='')

    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--use_normalized', action='store_true')
    parser.add_argument('--expt_no', type=int, required=True)
    
    return parser.parse_args()

# %% ../nbs/37_training-msmarco-distilbert-from-scratch.ipynb 21
if __name__ == '__main__':

    input_args = parse_args()

    if input_args.use_normalized: 
        output_dir = '/home/aiscuser/scratch1/outputs/mogicX/37_training-msmarco-distilbert-from-scratch-002'
    else: 
        output_dir = f'/home/aiscuser/scratch1/outputs/mogicX/37_training-msmarco-distilbert-from-scratch-{input_args.expt_no:03d}'

    print("Model directory:", output_dir)

    if input_args.exact: 
        raise ValueError("Arguement 'exact' is not allowed.")
    
    if not input_args.only_test:
        raise ValueError("Arguement 'only_test' required.")
    
    config_file = '/data/datasets/msmarco/XC/configs/data.json'
    config_key = 'data'
    
    mname = 'distilbert-base-uncased'

    pkl_file = get_pkl_file(input_args.pickle_dir, 'msmarco_data_distilbert-base-uncased', input_args.use_sxc_sampler, 
                            input_args.exact, input_args.only_test)

    do_inference = input_args.do_train_inference or input_args.do_test_inference or input_args.save_train_prediction or input_args.save_test_prediction or input_args.save_representation

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, 
                        only_test=input_args.only_test, n_slbl_samples=1)

    args = XCLearningArguments(
        output_dir=output_dir,
        logging_first_step=True,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=1600,
        representation_num_beams=200,
        representation_accumulation_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=5,
        num_train_epochs=300,
        predict_with_representation=True,
        representation_search_type='BRUTEFORCE',
        adam_epsilon=1e-6,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        label_names=['plbl2data_idx', 'plbl2data_data2ptr'],
    
        group_by_cluster=True,
        num_clustering_warmup_epochs=10,
        num_cluster_update_epochs=5,
        num_cluster_size_update_epochs=25,
        clustering_type='EXPO',
        minimum_cluster_size=2,
        maximum_cluster_size=1600,
    
        metric_for_best_model='P@1',
        load_best_model_at_end=True,
        target_indices_key='plbl2data_idx',
        target_pointer_key='plbl2data_data2ptr',
    
        use_encoder_parallel=True,
        max_grad_norm=None,
        fp16=True,

        use_cpu_for_searching=True,
        use_cpu_for_clustering=True,
    )

    def model_fn(mname):
        model = DBT023.from_pretrained(mname, normalize=input_args.use_normalized, use_layer_norm=True, use_encoder_parallel=True)
        return model
    
    def init_fn(model): 
        model.init_dr_head()

    metric = PrecReclMrr(block.test.dset.n_lbl, block.test.data_lbl_filterer, pk=10, rk=200, rep_pk=[1, 3, 5, 10], 
                         rep_rk=[10, 100, 200], mk=[5, 10, 20])

    bsz = max(args.per_device_train_batch_size, args.per_device_eval_batch_size)*torch.cuda.device_count()

    model = load_model(args.output_dir, model_fn, {"mname": mname}, init_fn, do_inference=do_inference, 
                       use_pretrained=input_args.use_pretrained)
    
    learn = XCLearner(
        model=model,
        args=args,
        train_dataset=block.train.dset if block.train else block.test.dset,
        eval_dataset=block.test.dset,
        data_collator=block.collator,
        compute_metrics=metric,
    )
    
    main(learn, input_args, n_lbl=block.test.dset.n_lbl)
    
