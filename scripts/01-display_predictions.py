import os, torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse
from typing import Union

from xcai.sdata import SXCDataset
from xcai.data import MainXCDataset, XCDataset
from xcai.basics import *
from xcai.analysis import *

def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    input_args = parse_args()
    extra_args = additional_args()

    input_args.text_mode = True
    input_args.use_sxc_sampler = True
    input_args.pickle_dir = '/home/aiscuser/scratch1/datasets/processed/' 

    extra_args.output_dir = '/data/outputs/mogicX/47_msmarco-gpt-category-linker-006'

    # extra_args.config_file = '/data/datasets/msmarco/XC/configs/data_lbl_gpt-category-conflated.json'
    extra_args.config_file = '/data/datasets/msmarco/XC/configs/data_lbl_gpt-category-linker-conflated-001.json'
    # extra_args.config_file = '/data/datasets/msmarco/XC/configs/data_lbl_gpt-category-linker-conflated-001-conflated-001.json'

    TYPE = 'predictions'

    if TYPE == 'dataset':
        output_dir = extra_args.output_dir 
        config_file = extra_args.config_file 
        config_key, fname = get_config_key(config_file)

        pkl_file = get_pkl_file(input_args.pickle_dir, f'msmarco_{fname}_distilbert-base-uncased', input_args.use_sxc_sampler, 
                                input_args.exact, input_args.only_test, use_text_mode=input_args.text_mode)

        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test, 
                n_slbl_samples=1, main_oversample=False, return_scores=True, use_tokenizer=not input_args.text_mode)

        disp_block = TextDataset(block.test.dset, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')

        np.random.seed(100)

        idxs = np.random.permutation(block.test.dset.n_data)[:100]
        fname = os.path.basename(config_file[:-5])
        disp_block.dump_txt(f'./outputs/msmarco_{fname}.txt', idxs)

    elif TYPE == 'predictions':
        output_dir = extra_args.output_dir 
        config_file = extra_args.config_file 
        config_key, fname = get_config_key(config_file)

        pkl_file = get_pkl_file(input_args.pickle_dir, f'msmarco_{fname}_distilbert-base-uncased', input_args.use_sxc_sampler, 
                                input_args.exact, input_args.only_test, use_text_mode=input_args.text_mode)

        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test, 
                n_slbl_samples=1, main_oversample=False, return_scores=True, use_tokenizer=not input_args.text_mode)

        pred_lbl = sp.load_npz(f'{output_dir}/predictions/test_predictions.npz')
        pred_block = get_pred_meta_dset(pred_lbl, block.test.dset, 'lnk_meta', meta_prefix='pred')

        disp_block = TextDataset(pred_block, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')

        np.random.seed(100)
        idxs = np.random.permutation(block.test.dset.n_data)[:100]
        fname = os.path.basename(output_dir)
        disp_block.dump_txt(f'./outputs/{fname}.txt', idxs)

