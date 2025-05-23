# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/20_classifier-for-amazontitles.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/20_classifier-for-amazontitles.ipynb 2
import os,torch, torch.multiprocessing as mp, pickle, numpy as np
from transformers import DistilBertConfig

from xcai.basics import *
from xcai.models.classifiers import CLS001
from xcai.models.distillation import DTL006, TCH001

# %% ../nbs/20_classifier-for-amazontitles.ipynb 4
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['WANDB_PROJECT'] = 'medic_03-amazontitles-classifier'

# %% ../nbs/20_classifier-for-amazontitles.ipynb 6
if __name__ == '__main__':
    output_dir = '/home/aiscuser/scratch1/outputs/mogicX/20_classifier-for-amazontitles-003'

    data_dir = '/data/datasets/benchmarks/G_Datasets/'
    config_file = 'amazontitles'
    config_key = 'data'

    teacher_dir = '/home/aiscuser/scratch1/outputs/mogicX/17_ngame-oracle-for-amazontitles-002/'
    classifier_dir = '/home/aiscuser/scratch1/outputs/mogicX/18_momos-for-amazontitles-001/' 
    
    mname = 'sentence-transformers/msmarco-distilbert-base-v4'
    
    input_args = parse_args()
    
    pkl_file = f'{input_args.pickle_dir}/mogicX/amazontitles_data_distilbert-base-uncased'
    pkl_file = f'{pkl_file}_sxc' if input_args.use_sxc_sampler else f'{pkl_file}_xcs'
    if input_args.only_test: pkl_file = f'{pkl_file}_only-test'
    pkl_file = f'{pkl_file}.joblib'

    do_inference = input_args.do_train_inference or input_args.do_test_inference or input_args.save_train_prediction or input_args.save_test_prediction or input_args.save_representation

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test,
                        data_dir=data_dir)

    """ Training arguements """
    args = XCLearningArguments(
        output_dir=output_dir,
        logging_first_step=True,
        per_device_train_batch_size=3072,
        per_device_eval_batch_size=1600,
        representation_num_beams=200,
        representation_accumulation_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=5,
        num_train_epochs=300,
        predict_with_representation=True,
        representation_search_type='BRUTEFORCE',
        adam_epsilon=1e-6,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-3,
        
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

        label_names=['lbl2data_idx', 'lbl2data_input_ids', 'lbl2data_attention_mask', 'lbl2data_data2ptr', 'plbl2data_idx', 'plbl2data_data2ptr'],

        use_cpu_for_searching=True,
        use_cpu_for_clustering=True,
    )
    """ Teacher model """
    m_teacher = TCH001.from_pretrained(f'{teacher_dir}/teacher', n_data=block.train.dset.n_data, n_lbl=block.n_lbl)
    m_teacher.freeze_embeddings()

    """ Classifiers """
    bsz = max(args.per_device_train_batch_size, args.per_device_eval_batch_size)*torch.cuda.device_count()
    
    m_student = CLS001.from_pretrained(f'{classifier_dir}/representation', n_train=block.train.dset.n_data, 
                                       n_test=block.test.dset.n_data, n_lbl=block.n_lbl, batch_size=bsz, 
                                       num_batch_labels=5000, margin=0.3, num_negatives=10, tau=0.1, apply_softmax=True)
    
    m_student.freeze_representation()
    m_student.init_lbl_embeddings()

    """ Distillation model """
    model = DTL006(DistilBertConfig(), m_student=m_student, m_teacher=m_teacher, bsz=bsz, tn_targ=5000, margin=0.3, tau=0.1, 
                   n_negatives=10, apply_softmax=True, teacher_data_student_label_loss_weight=0.1, 
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

    main(learn, input_args, n_lbl=block.n_lbl)
    
