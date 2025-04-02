# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/11_ngame-oracle-for-wikiseealsotitles-noise.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/11_ngame-oracle-for-wikiseealsotitles-noise.ipynb 3
import os,torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp

from xcai.basics import *
from xcai.models.PPP0XX import DBT009,DBT011

# %% ../nbs/11_ngame-oracle-for-wikiseealsotitles-noise.ipynb 5
os.environ['CUDA_VISIBLE_DEVICES'] = '8,9,10,11'
os.environ['WANDB_PROJECT'] = 'mogicX_01-wikiseealsotitles-oracle'

# %% ../nbs/11_ngame-oracle-for-wikiseealsotitles-noise.ipynb 19
if __name__ == '__main__':
    output_dir = '/home/aiscuser/scratch1/outputs/mogicX/11_ngame-oracle-for-wikiseealsotitles-noise'
    # output_dir = '/data/projects/xc_nlg/outputs/67-ngame-ep-for-wikiseealso-with-input-concatenation-6-3'

    data_dir = None
    config_file = '/home/aiscuser/scratch1/mogicX/configs/11_ngame-oracle-for-wikiseealsotitles-noise-002.json'
    config_key = 'data_category_linker'

    # data_dir = '/data/datasets/benchmarks/'
    # config_file = '/home/aiscuser/scratch1/mogicX/configs/12_momos-for-wikiseealsotitles-noise_data_category_ngame-linker.json'
    # config_key = 'data_category_linker'

    # data_dir = '/data/datasets/benchmarks/'
    # config_file = 'wikiseealsotitles'
    # config_key = 'data_lnk'

    mname = 'sentence-transformers/msmarco-distilbert-base-v4'

    meta_name = 'lnk'

    input_args = parse_args()

    pkl_file = f'{input_args.pickle_dir}/mogicX/11-ngame-oracle-for-wikiseealsotitles-noise-002_test'

    pkl_file = f'{pkl_file}_sxc' if input_args.use_sxc_sampler else f'{pkl_file}_xcs'
    if input_args.only_test: pkl_file = f'{pkl_file}_only-test'
    pkl_file = f'{pkl_file}.joblib'
    aug_file = pkl_file[:-7] + f'_aug{meta_name}-128.joblib'

    do_inference = input_args.do_train_inference or input_args.do_test_inference or input_args.save_train_prediction or input_args.save_test_prediction or input_args.save_representation

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)

    if os.path.exists(aug_file) and not input_args.build_block:
        block = joblib.load(aug_file)
    else:
        block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test,
                           sampling_features=[('lbl2data',1)], oversample=False, data_dir=data_dir)
        
        block = AugmentMetaInputIdsTfm.apply(block, f'{meta_name}_meta', 'data', 128, True)
        block = AugmentMetaInputIdsTfm.apply(block, f'{meta_name}_meta', 'lbl', 128, True)
        
        block.train.dset.data.data_info['input_ids'] = block.train.dset.data.data_info[f'input_ids_aug_{meta_name}']
        block.train.dset.data.data_info['attention_mask'] = block.train.dset.data.data_info[f'attention_mask_aug_{meta_name}']
        block.test.dset.data.data_info['input_ids'] = block.test.dset.data.data_info[f'input_ids_aug_{meta_name}']
        block.test.dset.data.data_info['attention_mask'] = block.test.dset.data.data_info[f'attention_mask_aug_{meta_name}']
        
        block.train.dset.data.lbl_info['input_ids'] = block.train.dset.data.lbl_info[f'input_ids_aug_{meta_name}']
        block.train.dset.data.lbl_info['attention_mask'] = block.train.dset.data.lbl_info[f'attention_mask_aug_{meta_name}']
        block.test.dset.data.lbl_info['input_ids'] = block.test.dset.data.lbl_info[f'input_ids_aug_{meta_name}']
        block.test.dset.data.lbl_info['attention_mask'] = block.test.dset.data.lbl_info[f'attention_mask_aug_{meta_name}']
        
        block.train.dset.meta = {}
        block.test.dset.meta = {}
    
        joblib.dump(block, aug_file)

    args = XCLearningArguments(
        output_dir=output_dir,
        logging_first_step=True,
        per_device_train_batch_size=600,
        per_device_eval_batch_size=600,
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
        adam_epsilon=1e-6,
        warmup_steps=100,
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

    metric = PrecReclMrr(block.n_lbl, block.test.data_lbl_filterer, prop=block.train.dset.data.data_lbl,
                         pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])
    
    bsz = max(args.per_device_train_batch_size, args.per_device_eval_batch_size)*torch.cuda.device_count()

    model = load_model(args.output_dir, model_fn, {"mname": mname, "bsz": bsz}, init_fn, do_inference=do_inference, 
            use_pretrained=input_args.use_pretrained)

    # # loading old model gives an error
    # from safetensors import safe_open
    # model_dir = f'{output_dir}/{os.path.basename(get_best_model(output_dir))}'

    # tensors = {}
    # with safe_open(f"{model_dir}/model.safetensors", framework="pt") as f:
    #     for k in f.keys():
    #         tensors[k] = f.get_tensor(k)

    # model.load_state_dict(tensors, strict=False)
    # # debug
    
    learn = XCLearner(
        model=model,
        args=args,
        train_dataset=block.train.dset,
        eval_dataset=block.test.dset,
        data_collator=block.collator,
        compute_metrics=metric,
    )
    
    main(learn, input_args, n_lbl=block.n_lbl)

    # main(learn, input_args, n_lbl=block.n_lbl, save_teacher=True)
