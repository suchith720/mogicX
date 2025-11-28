import os, torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse
from typing import Union, List
from tqdm.auto import tqdm

from xcai.core import *
from xcai.basics import *
from xcai.analysis import *
from xcai.sdata import SXCDataset
from xcai.data import MainXCDataset, XCDataset
from xcai.sdata import SMetaXCDataset, SXCDataset 

from xclib.utils.sparse import retain_topk


DATASETS = ['msmarco', 'arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa', 'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs',
        'scifact', 'webis-touche2020', 'trec-covid', 'cqadupstack/android', 'cqadupstack/english', 'cqadupstack/gaming', 'cqadupstack/gis',
        'cqadupstack/mathematica', 'cqadupstack/physics', 'cqadupstack/programmers', 'cqadupstack/stats', 'cqadupstack/tex', 'cqadupstack/unix',
        'cqadupstack/webmasters', 'cqadupstack/wordpress']


def get_pred_dset_for_msmarco(pred:sp.csr_matrix, meta_tuples:List, dset:Union[XCDataset,SXCDataset]):
    kwargs = {k: getattr(dset.data, k) for k in [o for o in vars(dset.data).keys() if not o.startswith('__')]}
    data = type(dset.data)(**kwargs)
    
    kwargs = {'prefix':'pred', 'data_meta': pred, 'meta_info': dset.data.lbl_info, 'return_scores': True}
    pred_dset = SMetaXCDataset(**kwargs) if isinstance(dset, SXCDataset) else MetaXCDataset(**kwargs)

    meta_kwargs = dict()
    for meta_name, meta, meta_info in meta_tuples:
        kwargs = {'prefix':meta_name, 'data_meta': meta, 'meta_info': meta_info, 'return_scores': True}
        meta_dset = SMetaXCDataset(**kwargs) if isinstance(dset, SXCDataset) else MetaXCDataset(**kwargs)
        meta_kwargs[f'{meta_name}_meta'] = meta_dset

    meta_kwargs['pred_meta'] = pred_dset
    
    return type(dset)(data, **meta_kwargs)


def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--topk', type=int, default=3) 
    parser.add_argument('--num', type=int, default=20) 
    parser.add_argument('--index_type', type=str, default='random')
    parser.add_argument('--use_train', action='store_true')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    input_args = parse_args()
    extra_args = additional_args()

    # Inputs arguements

    input_args.text_mode = True
    input_args.use_sxc_sampler = True
    input_args.pickle_dir = '/home/aiscuser/scratch1/datasets/processed/' 

    extra_args.output_dir ='/data/outputs/mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-002/'
    output_dir = extra_args.output_dir 

    meta_file = '/data/datasets/beir/msmarco/XC/raw_data/category-gpt_conflated.raw.csv'
    meta_info = Info.from_txt(meta_file, info_column_names=["identifier", "input_text"])
    
    info = np.load('/home/aiscuser/scratch1/tmp/info.npy')
    dset_len = np.load('/home/aiscuser/scratch1/tmp/dset_len.npy')

    DATASETS = ['msmarco']


    for i, dataset in tqdm(enumerate(DATASETS), total=len(DATASETS)):
        extra_args.config_file = f'configs/beir/{dataset}/{dataset.replace("/", "-")}_data-gpt-category-ngame-linker_conflated.json'
        config_file = extra_args.config_file 
        config_key, fname = get_config_key(config_file)

        # Load data

        dataset = dataset.replace('/', '-')
        pkl_file = get_pkl_file(input_args.pickle_dir, f'{dataset}_{fname}_distilbert-base-uncased', input_args.use_sxc_sampler, 
                                input_args.exact, input_args.only_test, use_text_mode=input_args.text_mode)

        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, 
                            only_test=input_args.only_test, n_slbl_samples=1, main_oversample=False, return_scores=True, 
                            use_tokenizer=not input_args.text_mode)

        if extra_args.use_train:
            pred = sp.load_npz(f'{output_dir}/predictions/47_msmarco-gpt-category-linker-002/train_predictions_{dataset}.npz')
            meta = sp.load_npz(f'/data/outputs/mogicX/47_msmarco-gpt-category-linker-002/predictions/train_predictions_{dataset}.npz')
        else:
            pred = sp.load_npz(f'{output_dir}/predictions/47_msmarco-gpt-category-linker-002/test_predictions_{dataset}.npz')
            fname = 'test_predictions.npz' if dataset == 'msmarco' else f'test_predictions_{dataset}.npz'
            meta = sp.load_npz(f'/data/outputs/mogicX/47_msmarco-gpt-category-linker-002/predictions/{fname}')

        meta = retain_topk(meta, k=5)
        pred = retain_topk(pred, k=extra_args.topk)

        dset = block.train.dset if extra_args.use_train else block.test.dset 

        if dataset == "msmarco":
            meta_gt = sp.load_npz(f'/data/datasets/beir/msmarco/XC/category-gpt_tst_X_Y_conflated.npz')
            pred_block = get_pred_dset_for_msmarco(pred, [('cat-pred', meta, meta_info), ('cat-gt', meta_gt, meta_info)], dset)
        else:
            pred_block = get_lbl_and_pred_dset_with_meta(pred, 'cat', meta, meta_info, dset)

        # Display predictions

        disp_block = TextDataset(pred_block, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')

        for index_type in ['random', 'good', 'bad']:
            extra_args.index_type = index_type 

            if extra_args.index_type == "random":
                ## Random indices

                np.random.seed(1000)
                idxs = np.random.permutation(block.test.dset.n_data)[:extra_args.num]
            else:
                ## Score based

                ptr = np.hstack([np.zeros(1, dtype=np.int64),np.cumsum(dset_len)])
                dset_info = info[ptr[i]:ptr[i+1]]

                idxs = np.argsort(dset_info[:, 0] - dset_info[:, 1])
                idxs = idxs[::-1] if extra_args.index_type == "good" else idxs
                idxs = idxs[:extra_args.num]

            example_dir = f'{output_dir}/examples/47_msmarco-gpt-category-linker-002/'
            os.makedirs(example_dir, exist_ok=True)
            fname = (
                f'{example_dir}/{dataset}_train_{extra_args.index_type}.json' 
                if extra_args.use_train else 
                f'{example_dir}/{dataset}_test_{extra_args.index_type}.json'
            )
            
            disp_block.dump(fname, idxs)

