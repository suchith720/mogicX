import os, torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse
from typing import Union

from xcai.sdata import SXCDataset
from xcai.data import MainXCDataset, XCDataset
from xcai.basics import *
from xcai.analysis import *

if __name__ == '__main__':
    output_dir = '/home/aiscuser/scratch1/outputs/mogicX/47_msmarco-gpt-category-linker-001'

    input_args = parse_args()

    # config_file = '/data/datasets/msmarco/XC/configs/data_gpt-category.json'
    config_file = '/data/datasets/msmarco/XC/configs/data_lbl_gpt-category_exact.json'
    config_key, fname = get_config_key(config_file)

    pkl_file = get_pkl_file(input_args.pickle_dir, f'msmarco_{fname}_distilbert-base-uncased', input_args.use_sxc_sampler, 
                            input_args.exact, input_args.only_test, use_text_mode=input_args.text_mode)

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test, 
            n_slbl_samples=1, main_oversample=False, return_scores=True, use_tokenizer=not input_args.text_mode)

    pred_lbl = sp.load_npz(f'{output_dir}/predictions/test_predictions.npz')
    # pred_block = get_pred_dset(pred_lbl, block.test.dset)
    pred_block = get_pred_meta_dset(pred_lbl, block.test.dset, 'cat_meta', meta_prefix='lnk')

    disp_block = TextDataset(pred_block, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')
    os.makedirs(f'{output_dir}/examples', exist_ok=True)

    idxs = np.random.permutation(block.test.dset.n_data)[:100]
    # disp_block.dump_txt(f'{output_dir}/examples/test_prediction.txt', idxs)
    disp_block.dump_txt(f'./outputs/47_msmarco-gpt-category-linker-001.txt', idxs)
