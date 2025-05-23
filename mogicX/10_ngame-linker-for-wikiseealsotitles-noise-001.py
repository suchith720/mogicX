# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_ngame-linker-for-wikiseealsotitles-noise.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/10_ngame-linker-for-wikiseealsotitles-noise.ipynb 3
import os,torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp

from xcai.basics import *
from xcai.models.PPP0XX import DBT009,DBT011

# %% ../nbs/10_ngame-linker-for-wikiseealsotitles-noise.ipynb 5
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['WANDB_PROJECT'] = 'mogicX_01-wikiseealsotitles-linker'

# %% ../nbs/10_ngame-linker-for-wikiseealsotitles-noise.ipynb 17
if __name__ == '__main__':
    # output_dir = '/home/aiscuser/scratch1/outputs/mogicX/10_ngame-linker-for-wikiseealsotitles-noise-001'
    output_dir = '/data/outputs/mogicX/10_ngame-linker-for-wikiseealsotitles-noise-001'

    config_file = '/data/datasets/benchmarks/(mapped)LF-WikiSeeAlsoTitles-320K/configs/data_category_noise-050.json'
    config_key = 'data_category'

    mname = 'sentence-transformers/msmarco-distilbert-base-v4'

    input_args = parse_args()

    pkl_file = f'{input_args.pickle_dir}/mogicX/wikiseealsotitles-noise_data-category_distilbert-base-uncased'
    pkl_file = f'{pkl_file}_sxc' if input_args.use_sxc_sampler else f'{pkl_file}_xcs'
    if input_args.only_test: pkl_file = f'{pkl_file}_only-test'
    pkl_file = f'{pkl_file}.joblib'

    do_inference = input_args.do_train_inference or input_args.do_test_inference or input_args.save_train_prediction or input_args.save_test_prediction or input_args.save_representation

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test)

    #debug
    # import xclib.data.data_utils as du
    # data_dir = '/data/datasets/benchmarks/(mapped)LF-WikiSeeAlsoTitles-320K/'
    # trn_meta = du.read_sparse_file(f'{data_dir}/category_trn_X_Y.txt')
    # tst_meta = du.read_sparse_file(f'{data_dir}/category_tst_X_Y.txt')

    # block.train.dset.meta['cat_meta'].data_meta = trn_meta
    # block.test.dset.meta['cat_meta'].data_meta = tst_meta
    #debug

    linker_block = block.linker_dset('cat_meta', remove_empty=False)

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

    metric = PrecReclMrr(linker_block.n_lbl, linker_block.test.data_lbl_filterer, prop=linker_block.train.dset.data.data_lbl,
                         pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])
    
    bsz = max(args.per_device_train_batch_size, args.per_device_eval_batch_size)*torch.cuda.device_count()

    model = load_model(args.output_dir, model_fn, {"mname": mname, "bsz": bsz}, init_fn, do_inference=do_inference, use_pretrained=input_args.use_pretrained)
    
    learn = XCLearner(
        model=model,
        args=args,
        train_dataset=linker_block.train.dset,
        eval_dataset=linker_block.test.dset,
        data_collator=linker_block.collator,
        compute_metrics=metric,
    )
    
    breakpoint()
    main(learn, input_args, n_lbl=linker_block.n_lbl, save_classifier=True)
    
    # dset = linker_block.test.dset.data
    # eval_dset = block.inference_dset(dset.data_info, dset.data_lbl, dset.lbl_info, dset.data_lbl_filterer)

    # dset = linker_block.train.dset.data
    # train_dset = block.inference_dset(dset.data_info, dset.data_lbl, dset.lbl_info, dset.data_lbl_filterer)
    # 
    # main(learn, input_args, n_lbl=linker_block.n_lbl, eval_dataset=eval_dset, train_dataset=train_dset, eval_k=20, train_k=20)

    # suffix = f'_{input_args.prediction_suffix}' if len(input_args.prediction_suffix) else ''
    # lbl_file = f'{args.output_dir}/predictions/label_predictions{suffix}.npz'
    # sp.save_npz(lbl_file, sp.csr_matrix((block.n_lbl, linker_block.n_lbl), dtype=np.float32))
