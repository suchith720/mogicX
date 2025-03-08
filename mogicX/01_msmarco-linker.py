# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_msmarco-linker.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/01_msmarco-linker.ipynb 3
import os,torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp

from xcai.basics import *
from xcai.models.PPP0XX import DBT009,DBT011

# %% ../nbs/01_msmarco-linker.ipynb 5
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['WANDB_PROJECT'] = 'mogicX_00-msmarco'

# %% ../nbs/01_msmarco-linker.ipynb 20
if __name__ == '__main__':
    output_dir = '/scratch/scai/phd/aiz218323/outputs/mogicX/01_msmarco-linker'

    config_file = '/scratch/scai/phd/aiz218323/datasets/msmarco/XC/configs/entity_gpt_exact.json'
    config_key = 'data_entity-gpt_exact'
    
    mname = 'sentence-transformers/msmarco-distilbert-dot-v5'

    input_args = parse_args()

    pkl_file = f'{input_args.pickle_dir}/mogicX/msmarco_data_distilbert-base-uncased'
    pkl_file = f'{pkl_file}_sxc' if input_args.use_sxc_sampler else f'{pkl_file}_xcs'
    
    if input_args.only_test: pkl_file = f'{pkl_file}_only-test'
    pkl_file = f'{pkl_file}_exact'
    
    pkl_file = f'{pkl_file}.joblib'

    do_inference = input_args.do_train_inference or input_args.do_test_inference or input_args.save_train_prediction or input_args.save_test_prediction or input_args.save_representation

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block)

    args = XCLearningArguments(
        output_dir=output_dir,
        logging_first_step=True,
        per_device_train_batch_size=800,
        per_device_eval_batch_size=800,
        representation_num_beams=200,
        representation_accumulation_steps=10,
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=5,
        num_train_epochs=300,
        predict_with_representation=True,
        representation_search_type='BRUTEFORCE',
        adam_epsilon=1e-6,                                                                                                                                          warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-4,
    
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
    )

    def model_fn(mname, bsz):
        model = DBT009.from_pretrained(mname, bsz=bsz, tn_targ=5000, margin=0.3, tau=0.1, n_negatives=10, 
                                       apply_softmax=True, use_encoder_parallel=True)
        return model
    
    def init_fn(model): 
        model.init_dr_head()

    linker_block = block.linker_dset('ent_meta')

    metric = PrecReclMrr(linker_block.n_lbl, linker_block.test.data_lbl_filterer, prop=linker_block.train.dset.data.data_lbl,
                         pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])

    bsz = max(args.per_device_train_batch_size, args.per_device_eval_batch_size)*torch.cuda.device_count()

    model = load_model(args.output_dir, model_fn, {"mname": mname, "bsz": bsz}, init_fn, do_inference=do_inference, use_pretrained=input_args.use_pretrained)
    
    learn = XCLearner(
        model=model,
        args=args,
        train_dataset=block.train.dset,
        eval_dataset=block.test.dset,
        data_collator=block.collator,
        compute_metrics=metric,
    )
    
    main(learn, input_args, n_lbl=linker_block.n_lbl)
    
