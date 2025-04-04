# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/31_momos-for-wikiseealsotitles.ipynb.

# %% auto 0
__all__ = ['get_label_representation', 'get_label_remap']

# %% ../nbs/31_momos-for-wikiseealsotitles.ipynb 2
import os,torch, torch.multiprocessing as mp, pickle, numpy as np
from transformers import DistilBertConfig

from xcai.main import *
from xcai.basics import *
from xcai.models.oak import OAK008
from xcai.models.distillation import DTL004,TCH001,TCH002
from xcai.clustering.cluster import BalancedClusters, get_cluster_size

from xclib.utils.sparse import retain_topk

from fastcore.utils import *

# %% ../nbs/31_momos-for-wikiseealsotitles.ipynb 4
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['WANDB_PROJECT'] = 'mogicX_01-wikiseealsotitles'

# %% ../nbs/31_momos-for-wikiseealsotitles.ipynb 6
@patch
def get_label_representation(
    cls:DTL004,
    data_idx:Optional[torch.Tensor]=None,
    data_input_ids:Optional[torch.Tensor]=None,
    data_attention_mask:Optional[torch.Tensor]=None,
    **kwargs
):
    return cls.m_student.get_label_representation(data_idx, data_input_ids, data_attention_mask, **kwargs)
    

# %% ../nbs/31_momos-for-wikiseealsotitles.ipynb 7
def get_label_remap(lbl_repr:torch.Tensor, cluster_sz:int=3):
    clusters = BalancedClusters.proc(lbl_repr.half(), min_cluster_sz=cluster_sz)

    lbl_remap = torch.zeros(lbl_repr.shape[0], dtype=torch.int64)
    for i,o in enumerate(clusters): lbl_remap[o] = i

    return lbl_remap, len(clusters)

# %% ../nbs/31_momos-for-wikiseealsotitles.ipynb 8
if __name__ == '__main__':
    output_dir = '/home/aiscuser/scratch1/outputs/mogicX/12_momos-for-wikiseealsotitles-noise-001'

    data_dir = '/data/datasets/benchmarks/'
    config_file = '/home/aiscuser/scratch1/mogicX/configs/12_momos-for-wikiseealsotitles-noise_data_category_ngame-linker.json'
    config_key = 'data_category_linker'

    model_output = '/home/aiscuser/scratch1/outputs/mogicX/11_ngame-oracle-for-wikiseealsotitles-noise/'
    meta_embed_file = '/data/datasets/ogb_weights/LF-WikiSeeAlsoTitles-320K/emb_weights.npy'
    
    meta_name = 'lnk'

    input_args = parse_args()

    pkl_file = f'{input_args.pickle_dir}/mogicX/wikiseealsotitles-noise_data-category-linker_distilbert-base-uncased'
    pkl_file = f'{pkl_file}_sxc' if input_args.use_sxc_sampler else f'{pkl_file}_xcs'
    if input_args.only_test: pkl_file = f'{pkl_file}_only-test'
    pkl_file = f'{pkl_file}.joblib'

    do_inference = input_args.do_train_inference or input_args.do_test_inference or input_args.save_train_prediction or input_args.save_test_prediction or input_args.save_representation

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test, 
                        n_slbl_samples=4, main_oversample=False, n_sdata_meta_samples=3, meta_oversample=False, train_meta_topk=5, test_meta_topk=3, 
                        data_dir=data_dir)

    """ Training arguements """
    args = XCLearningArguments(
        output_dir=output_dir,
        logging_first_step=True,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        representation_num_beams=200,
        representation_accumulation_steps=10,
        save_strategy="steps",
        evaluation_strategy="steps",
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
        
        # label_names=['lbl2data_idx', 'lbl2data_input_ids', 'lbl2data_attention_mask', 'lnk2data_idx'],
        label_names=['lbl2data_idx', 'lbl2data_input_ids', 'lbl2data_attention_mask', 'lbl2data_data2ptr', 'plbl2data_idx', 'plbl2data_data2ptr',
                 f'{meta_name}2data_idx', f'{meta_name}2data_input_ids', f'{meta_name}2data_attention_mask', f'{meta_name}2data_data2ptr'],
        
        prune_metadata=False,
        num_metadata_prune_warmup_epochs=10,
        num_metadata_prune_epochs=5,
        metadata_prune_batch_size=2048,
        prune_metadata_names=['lnk_meta'],
        use_data_metadata_for_pruning=True,
    
        predict_with_augmentation=False,
        use_augmentation_index_representation=True,
    
        data_aug_meta_name='lnk',
        augmentation_num_beams=None,
        data_aug_prefix='lnk',
        use_label_metadata=False,
        
        data_meta_batch_size=2048,
        augment_metadata=False,
        num_metadata_augment_warmup_epochs=10,
        num_metadata_augment_epochs=5,
    
        use_cpu_for_searching=True,
        use_cpu_for_clustering=True,
    )

    """ Teacher model """
    m_teacher = TCH001.from_pretrained(f'{model_output}/teacher', n_data=block.train.dset.n_data, n_lbl=block.n_lbl)
    m_teacher.freeze_embeddings()

    if do_inference:
        n_clusters = get_cluster_size(m_teacher.lbl_repr.weight.shape[0], cluster_sz=3)
    else:
        lbl_remap, n_clusters = get_label_remap(m_teacher.lbl_repr.weight, cluster_sz=3)

    """ Student model """
    bsz = max(args.per_device_train_batch_size, args.per_device_eval_batch_size)*torch.cuda.device_count()

    m_student = OAK008.from_pretrained('sentence-transformers/msmarco-distilbert-base-v4', batch_size=bsz, num_batch_labels=5000,
                                       margin=0.3, num_negatives=10, tau=0.1, apply_softmax=True,
                                       
                                       data_aug_meta_prefix='lnk2data', lbl2data_aug_meta_prefix=None,
                                       data_pred_meta_prefix=None, lbl2data_pred_meta_prefix=None,
                                       
                                       num_metadata=block.train.dset.meta['lnk_meta'].n_meta, resize_length=5000,
                                       n_clusters=n_clusters, n_labels=block.n_lbl,
                                       
                                       calib_margin=0.05, calib_num_negatives=10, calib_tau=0.1, calib_apply_softmax=False,
                                       calib_loss_weight=0.1, use_calib_loss=True,
                                       
                                       use_query_loss=True,
                                       
                                       meta_loss_weight=0.0,
                                       
                                       fusion_loss_weight=0.1, use_fusion_loss=False,
                                       
                                       use_encoder_parallel=True)
    if not do_inference:
        m_student.init_retrieval_head()
        m_student.init_cross_head()
        m_student.init_meta_embeddings()
        m_student.init_label_embeddings()
        m_student.set_label_remap(lbl_remap)
        
        meta_embeddings = np.load(meta_embed_file)
        m_student.encoder.set_pretrained_meta_embeddings(torch.tensor(meta_embeddings, dtype=torch.float32))
        m_student.encoder.freeze_pretrained_meta_embeddings()

    """ Distillation model """
    if do_inference:
        mname = f'{output_dir}/{os.path.basename(get_best_model(output_dir))}'
        model = DTL004.from_pretrained(mname, m_student=m_student, m_teacher=m_teacher, bsz=bsz, tn_targ=5000, margin=0.3, tau=0.1, 
                n_negatives=10, apply_softmax=True, teacher_data_student_label_loss_weight=1.0, student_data_teacher_label_loss_weight=0.0, 
                data_mse_loss_weight=0.1, label_mse_loss_weight=0.0)
    else:
        model = DTL004(DistilBertConfig(), m_student=m_student, m_teacher=m_teacher, bsz=bsz, tn_targ=5000, margin=0.3, tau=0.1, 
                       n_negatives=10, apply_softmax=True, teacher_data_student_label_loss_weight=1.0, 
                       student_data_teacher_label_loss_weight=0.0, data_mse_loss_weight=0.1, label_mse_loss_weight=0.0)
    
    """ Training """
    metric = PrecRecl(block.n_lbl, block.test.data_lbl_filterer, prop=block.train.dset.data.data_lbl,
                      pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200])

    learn = XCLearner(
        model=model, 
        args=args,
        train_dataset=block.train.dset,
        eval_dataset=block.test.dset,
        data_collator=block.collator,
        compute_metrics=metric,
    )

    if do_inference: os.environ['WANDB_MODE'] = 'disabled'
    
    main(learn, input_args, n_lbl=block.n_lbl, save_classifier=True)
    
