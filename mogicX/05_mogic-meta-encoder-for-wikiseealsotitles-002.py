# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_mogic-meta-encoder-for-wikiseealsotitles.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/05_mogic-meta-encoder-for-wikiseealsotitles.ipynb 3
import os,torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp
from transformers import DistilBertConfig

from xcai.main import *
from xcai.basics import *
from xcai.clustering.cluster import get_cluster_mapping, get_cluster_size

from xcai.models.oakY import OAK007
from xcai.models.distillation import DTL004,TCH001

# %% ../nbs/05_mogic-meta-encoder-for-wikiseealsotitles.ipynb 5
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['WANDB_PROJECT'] = 'mogicX_01-wikiseealsotitles'

# %% ../nbs/05_mogic-meta-encoder-for-wikiseealsotitles.ipynb 24
if __name__ == '__main__':
    # output_dir = '/home/aiscuser/scratch1/outputs/mogicX/05_mogic-meta-encoder-for-wikiseealsotitles-001'
    output_dir = '/data/outputs/mogic/12_momos-for-wikiseealsotitles-meta-encoder-002'

    config_file = '/home/aiscuser/scratch1/mogicX/configs/14_ngame-linker-for-wikiseealsotitles-001.json'
    config_key = 'data_category_linker'

    teacher_model = '/data/projects/xc_nlg/outputs/67-ngame-ep-for-wikiseealso-with-input-concatenation-6-3/teacher/'
    student_model = 'sentence-transformers/msmarco-distilbert-base-v4'

    meta_name = 'lnk'
    
    input_args = parse_args()

    pkl_file = f'{input_args.pickle_dir}/mogicX/wikiseealsotitles_data-ngame-lnk_distilbert-base-uncased'
    pkl_file = f'{pkl_file}_sxc' if input_args.use_sxc_sampler else f'{pkl_file}_xcs'
    if input_args.only_test: pkl_file = f'{pkl_file}_only-test'
    pkl_file = f'{pkl_file}.joblib'

    do_inference = input_args.do_train_inference or input_args.do_test_inference or input_args.save_train_prediction or input_args.save_test_prediction or input_args.save_representation

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, 
                        only_test=input_args.only_test, n_sdata_meta_samples=3, meta_oversample=False, test_meta_topk=3)

    # config_file = '/home/aiscuser/scratch1/mogicX/configs/14_ngame-linker-for-wikiseealsotitles-001_wikiseealsotitles-20250123-full.json'
    # config_file = '/home/aiscuser/scratch1/mogicX/configs/14_ngame-linker-for-wikiseealsotitles-001_wikiseealsotitles-20250123-metadata-20251203-full.json'
    # config_file = '/home/aiscuser/scratch1/mogicX/configs/14_ngame-linker-for-wikiseealsotitles-001_metadata-20250123-full.json'

    # fname = '07_ngame-linker-for-wikiseealsotitles-20250123-001_metadata-full'
    # fname = '07_ngame-linker-for-wikiseealsotitles-20250123-001_wikiseealsotitles-full'
    fname = '07_ngame-linker-for-wikiseealsotitles-20250123-001_wikiseealsotitles-metadata-full'
    config_file = f'/home/aiscuser/scratch1/mogicX/configs/{fname}.json'
    config_key = 'data_category_linker'

    pkl_file = f'{input_args.pickle_dir}/mogicX/wikiseealsotitles-20250123_{fname.replace("_", "-")}_distilbert-base-uncased'
    pkl_file = f'{pkl_file}_sxc' if input_args.use_sxc_sampler else f'{pkl_file}_xcs'
    if input_args.only_test: pkl_file = f'{pkl_file}_only-test'
    pkl_file = f'{pkl_file}.joblib'

    eval_block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, 
            only_test=input_args.only_test, n_sdata_meta_samples=3, meta_oversample=False, test_meta_topk=3)

    args = XCLearningArguments(
        output_dir=output_dir,
        logging_first_step=True,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        representation_num_beams=200,
        representation_accumulation_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=5,
        num_train_epochs=300,
        predict_with_representation=True,
        adam_epsilon=1e-6,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-4,
        representation_search_type='BRUTEFORCE',
    
        output_representation_attribute='data_fused_repr',
        label_representation_attribute='data_repr',
        metadata_representation_attribute='data_repr',
        data_augmentation_attribute='data_repr',
        representation_attribute='data_fused_repr',
        clustering_representation_attribute='data_fused_repr',
    
        group_by_cluster=True,
        num_clustering_warmup_epochs=10,
        num_cluster_update_epochs=5,
        num_cluster_size_update_epochs=25,
        use_data_metadata_for_clustering=True,
        clustering_type='EXPO',
        minimum_cluster_size=2,
        maximum_cluster_size=1600,

        metric_for_best_model='P@1',
        load_best_model_at_end=True,
        target_indices_key='plbl2data_idx',
        target_pointer_key='plbl2data_data2ptr',
    
        use_distributional_representation=False,
        use_encoder_parallel=True,
        max_grad_norm=None,
        fp16=True,
    
        label_names=['lbl2data_idx', 'lbl2data_input_ids', 'lbl2data_attention_mask', 'lbl2data_data2ptr', 'plbl2data_idx', 'plbl2data_data2ptr', 
            f'{meta_name}2data_idx', f'{meta_name}2data_input_ids', f'{meta_name}2data_attention_mask', f'{meta_name}2data_data2ptr'],
    
        prune_metadata=False,
        num_metadata_prune_warmup_epochs=10,
        num_metadata_prune_epochs=5,
        metadata_prune_batch_size=2048,
        prune_metadata_names=[f'{meta_name}_meta'],
        use_data_metadata_for_pruning=True,
    
        predict_with_augmentation=False,
        use_augmentation_index_representation=True,
    
        data_aug_meta_name=meta_name,
        augmentation_num_beams=None,
        data_aug_prefix=meta_name,
        use_label_metadata=False,
    
        data_meta_batch_size=2048,
        augment_metadata=False,
        num_metadata_augment_warmup_epochs=10,
        num_metadata_augment_epochs=5,
    
        use_cpu_for_searching=True,
        use_cpu_for_clustering=True,
    )

    def model_fn(teacher_model, student_model, mname, do_inference, use_pretrained, bsz):
        m_teacher = TCH001.from_pretrained(teacher_model, n_data=block.train.dset.n_data, n_lbl=block.n_lbl)
        m_teacher.freeze_embeddings()
    
        if not do_inference or use_pretrained:
            cluster_sz = 3
            cluster_file = f'{teacher_model}/clusters_{cluster_sz:03d}.joblib'
            if os.path.exists(cluster_file): 
                label_cluster_mapping, n_clusters = joblib.load(cluster_file)
            else:
                label_cluster_mapping, n_clusters = get_cluster_mapping(m_teacher.lbl_repr.weight, cluster_sz=3)
                joblib.dump((label_cluster_mapping, n_clusters), cluster_file)
        else:
            n_clusters = get_cluster_size(m_teacher.lbl_repr.weight.shape[0], cluster_sz=3)
        
        m_student = OAK007.from_pretrained(student_model, batch_size=bsz, num_batch_labels=5000, margin=0.3, 
                                           num_negatives=10, tau=0.1, apply_softmax=True,
                                           
                                           data_aug_meta_prefix=f'{meta_name}2data', lbl2data_aug_meta_prefix=None, 
                                           data_pred_meta_prefix=None, lbl2data_pred_meta_prefix=None,
                                           
                                           calib_margin=0.05, calib_num_negatives=10, calib_tau=0.1, calib_apply_softmax=False, 
                                           calib_loss_weight=0.1, use_calib_loss=False,
                                           
                                           n_labels=block.n_lbl, n_clusters=n_clusters, use_query_loss=True, use_encoder_parallel=True)
    
        if not do_inference or use_pretrained:
            model = DTL004(DistilBertConfig(), m_student=m_student, m_teacher=m_teacher, bsz=bsz, tn_targ=5000, margin=0.3, tau=0.1,
                           n_negatives=10, apply_softmax=True, teacher_data_student_label_loss_weight=1.0,student_data_teacher_label_loss_weight=0.0, 
                           data_mse_loss_weight=0.1, label_mse_loss_weight=0.0)
            model.m_student.set_label_cluster_mapping(label_cluster_mapping)
        else:
            model = DTL004.from_pretrained(mname, m_student=m_student, m_teacher=m_teacher, bsz=bsz, tn_targ=5000, margin=0.3, tau=0.1,
                                           n_negatives=10, apply_softmax=True, teacher_data_student_label_loss_weight=1.0,
                                           student_data_teacher_label_loss_weight=0.0, data_mse_loss_weight=0.1, label_mse_loss_weight=0.0)
        return model
    
    def init_fn(model):
        model.m_student.init_retrieval_head()
        model.m_student.init_cross_head()
        model.m_student.init_meta_encoder()
        model.m_student.init_label_embeddings()

    bsz = max(args.per_device_train_batch_size, args.per_device_eval_batch_size)*torch.cuda.device_count()

    model = load_model(args.output_dir, model_fn, {'teacher_model': teacher_model, 'student_model': student_model, 'mname': None, 
                                                   'do_inference': do_inference, 'use_pretrained': input_args.use_pretrained, 'bsz': bsz}, 
                       init_fn, do_inference=do_inference, use_pretrained=input_args.use_pretrained)

    # metric = PrecReclMrr(block.n_lbl, block.test.data_lbl_filterer, prop=block.train.dset.data.data_lbl,
    #                      pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])
    # learn = XCLearner(
    #     model=model.m_student,
    #     args=args,
    #     train_dataset=block.train.dset,
    #     eval_dataset=block.test.dset,
    #     data_collator=block.collator,
    #     compute_metrics=metric,
    # )
    # main(learn, input_args, n_lbl=block.n_lbl)

    # map_file = f'{output_dir}/predictions/wikiseealsotitles-20250123-label_mapping_final.npy'
    # old2new_label_mapping = np.load(map_file)

    # with torch.no_grad():
    #     model.m_student.label_cluster_mapping = model.m_student.label_cluster_mapping[old2new_label_mapping]

    metric = PrecReclMrr(eval_block.n_lbl, eval_block.test.data_lbl_filterer, prop=eval_block.train.dset.data.data_lbl,
                         pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])
    learn = XCLearner(
        model=model.m_student,
        args=args,
        train_dataset=eval_block.train.dset,
        eval_dataset=eval_block.test.dset,
        data_collator=eval_block.collator,
        compute_metrics=metric,
    )
    o = main(learn, input_args, n_lbl=eval_block.n_lbl)

    print(o[4].shape)
