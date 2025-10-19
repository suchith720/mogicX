import os, torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse
from typing import Union

from xcai.basics import *
from xcai.analysis import *
from xcai.sdata import SXCDataset
from xcai.data import MainXCDataset, XCDataset
from xcai.sdata import SMetaXCDataset, SXCDataset 

from xclib.utils.sparse import retain_topk

def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--meta_name', type=str, default=None)

    parser.add_argument('--iteration_num', type=int)
    parser.add_argument('--iteration_preds', action='store_true')
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    input_args = parse_args()
    extra_args = additional_args()

    input_args.text_mode = True
    input_args.use_sxc_sampler = True
    input_args.pickle_dir = '/home/aiscuser/scratch1/datasets/processed/' 

    input_args.dataset = 'all-beir'
    extra_args.output_dir = f'/data/datasets/beir/{input_args.dataset}/XC'
    extra_args.config_file = f'{extra_args.output_dir}/configs/data.json'

    TYPE = 'dataset'

    if TYPE == 'dataset':
        output_dir = extra_args.output_dir 

        config_file = extra_args.config_file 
        config_key, fname = get_config_key(config_file)

        pkl_file = get_pkl_file(input_args.pickle_dir, f'{input_args.dataset}_{fname}_distilbert-base-uncased', input_args.use_sxc_sampler, 
                                input_args.exact, input_args.only_test, use_text_mode=input_args.text_mode)

        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test, 
                            n_slbl_samples=1, main_oversample=False, return_scores=True, use_tokenizer=not input_args.text_mode)

        disp_block = TextDataset(block.test.dset, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')

        np.random.seed(100)

        idxs = np.random.permutation(block.test.dset.n_data)[:100]

        fname = os.path.basename(config_file[:-5])
        os.makedirs(f'{output_dir}/examples', exist_ok=True)
        disp_block.dump_txt(f'{output_dir}/examples/{fname}.txt', idxs)

    elif TYPE == 'predictions':
        output_dir = extra_args.output_dir 

        config_file = extra_args.config_file 
        config_key, fname = get_config_key(config_file)

        pkl_file = get_pkl_file(input_args.pickle_dir, f'{input_args.dataset}_{fname}_distilbert-base-uncased', input_args.use_sxc_sampler, 
                                input_args.exact, input_args.only_test, use_text_mode=input_args.text_mode)

        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test, 
                            n_slbl_samples=1, main_oversample=False, return_scores=True, use_tokenizer=not input_args.text_mode)

        pred = sp.load_npz(f'{output_dir}/predictions/test_predictions.npz')

        if extra_args.meta_name is None:
            pred_block = get_pred_meta_dset(pred, block.test.dset, extra_args.meta_name, meta_prefix='pred')
        else:
            pred_block = get_pred_dset(pred, block.test.dset)

        disp_block = TextDataset(pred_block, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')

        np.random.seed(100)
        idxs = np.random.permutation(block.test.dset.n_data)[:100]
        fname = os.path.basename(output_dir[:-1] if output_dir[-1] == '/' else output_dir) 
        disp_block.dump_txt(f'{output_dir}/examples.txt', idxs)

    elif TYPE == 'iterative':
        input_args.text_mode = False
        input_args.only_test = True
        output_dir = extra_args.output_dir 

        config_file = extra_args.config_file 
        config_key, fname = get_config_key(config_file)
        mname = 'distilbert-base-uncased'

        pkl_file = get_pkl_file(input_args.pickle_dir, f'msmarco_{fname}_distilbert-base-uncased', input_args.use_sxc_sampler, 
                                input_args.exact, input_args.only_test, use_text_mode=input_args.text_mode)

        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, 
                            only_test=input_args.only_test, main_oversample=True, meta_oversample=True, return_scores=True, 
                            n_slbl_samples=1, n_sdata_meta_samples=1)

        # Iterative inference experiment
        iteration_dir = '/data/share/from_deepak/iterative/44_distilbert-gpt-category-linker-oracle-for-msmarco-005'

        if extra_args.iteration_num > 0:
            fname = f'{iteration_dir}/iter_{extra_args.iteration_num}/raw_data/test_category-gpt-linker.raw.csv'
            data_info = Info.from_txt(fname, max_sequence_length=300, padding=True, return_tensors='pt', info_column_names=["identifier", "input_text"], 
                                      tokenization_column="input_text", use_tokenizer=True, tokenizer=mname)
            block.test.dset.data.data_info = data_info
        # Iterative inference experiment

        iteration_dir = iteration_dir[:-1] if iteration_dir[-1] == "/" else iteration_dir
        save_dir = f'./outputs/{os.path.basename(iteration_dir)}/'
        os.makedirs(save_dir, exist_ok=True)

        if extra_args.iteration_preds:
            pred_file = (
                f'{output_dir}/predictions/test_predictions.npz' 
                if extra_args.iteration_num == 0 else 
                f'{output_dir}/predictions/test_predictions_iter-{extra_args.iteration_num+1}.npz'
            )
            pred_lbl = retain_topk(sp.load_npz(pred_file), k=5)

            meta_block = SMetaXCDataset(prefix='pred', data_meta=pred_lbl, meta_info=block.test.dset.data.lbl_info, return_scores=True)
            pred_block = SXCDataset(block.test.dset.data, pred_meta=meta_block) 

            disp_block = TextDataset(pred_block, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')

            np.random.seed(100)
            idxs = np.random.permutation(block.test.dset.n_data)[:100]

            disp_block.dump_txt(f'{save_dir}/test_predictions_iter-{extra_args.iteration_num}.txt', idxs)

        else:
            disp_block = TextDataset(block.test.dset, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')

            np.random.seed(100)

            idxs = np.random.permutation(block.test.dset.n_data)[:100]
            fname = os.path.basename(config_file[:-5])

            disp_block.dump_txt(f'{save_dir}/msmarco_{fname}_iter-{extra_args.iteration_num}.txt', idxs)


