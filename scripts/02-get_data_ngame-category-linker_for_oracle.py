import scipy.sparse as sp, os, argparse, json

from typing import List
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

from xcai.main import *
from sugar.core import *


def get_data_category(data_cat:sp.csr_matrix, data_txt:List, cat_txt:List):
    assert data_cat.shape[0] == len(data_txt), f'{data_cat.shape[0]} != {len(data_txt)}'
    assert data_cat.shape[1] == len(cat_txt), f'{data_cat.shape[1]} != {len(cat_txt)}'

    data_cat_txt = []
    for data, txt in tqdm(zip(data_cat, data_txt), total=len(data_txt)):
        cats = [cat_txt[i] for i in data.indices]
        data_cat_txt.append(txt + " <CATEGORIES> " + " || ".join(cats))
    return data_cat_txt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--expt_no', type=int, required=True)
    parser.add_argument('--abs_thresh', type=float, default=None)
    parser.add_argument('--diff_thresh', type=float, default=None)
    parser.add_argument('--type', type=str, default="raw")
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    input_args = parse_args()

    TYPE = "prediction"

    output_dir = (
        f'/data/outputs/mogicX/47_msmarco-gpt-category-linker-{input_args.expt_no:03d}' 
        if TYPE == 'prediction' else 
        f'/data/datasets/{input_args.dataset}/XC/'
    )
    os.makedirs(f'{output_dir}/raw_data', exist_ok=True)

    prefix = 'category-gpt'
    meta_info_numbers = {
        0: f'{prefix}-linker',
        1: f'{prefix}', 
        2: f'{prefix}_conflated', 
        3: f'{prefix}_conflated-001', 
        4: f'{prefix}_conflated-002', 
        7: f'{prefix}-linker_conflated-001_conflated-001', 
        8: f'{prefix}-linker_conflated-001_conflated-001', 
        9: f'{prefix}-linker_conflated-001_conflated-001', 
    }

    def apply_threshold(mat, args):
        if input_args.abs_thresh is not None: 
            mat = Filter.threshold(mat, t=args.abs_thresh)
        if input_args.diff_thresh is not None: 
            mat = Filter.difference(mat, t=args.diff_thresh)
        return mat

    if input_args.type == "raw":
        # Load metadata
        cat_ids, cat_txt = load_raw_file(f'/data/datasets/msmarco/XC/raw_data/{meta_info_numbers[input_args.expt_no]}.raw.csv')
        
        # Train dataset
        if input_args.dataset == "msmarco":
            trn_ids, trn_txt = load_raw_file(f'/data/datasets/{input_args.dataset}/XC/raw_data/train.raw.txt')
            fname = (
                f'{output_dir}/predictions/train_predictions.npz' 
                if TYPE == 'prediction' else 
                f'{output_dir}/{meta_info_numbers[input_args.expt_no]}_trn_X_Y.npz'
            )
            trn_cat = retain_topk(sp.load_npz(fname), k=5)
            trn_cat = apply_threshold(trn_cat, input_args)

            trn_cat_txt = get_data_category(trn_cat, trn_txt, cat_txt)
            fname = (
                f'{output_dir}/raw_data/train_{meta_info_numbers[input_args.expt_no]}_{input_args.dataset}.raw.csv'
                if TYPE == 'prediction' else
                f'{output_dir}/raw_data/train_{meta_info_numbers[input_args.expt_no]}.raw.csv'
            )
            save_raw_file(fname, trn_ids, trn_cat_txt)
        
        # Test dataset
        fname = (
            (
                f'{output_dir}/predictions/test_predictions.npz' 
                if input_args.dataset == "msmarco" else 
                f'{output_dir}/predictions/test_predictions_{input_args.dataset}.npz'
            )
            if TYPE == 'prediction' else 
            f'{output_dir}/{meta_info_numbers[input_args.expt_no]}_tst_X_Y.npz'
        )
        if input_args.dataset == "msmarco":
            tst_ids, tst_txt = load_raw_file(f'/data/datasets/{input_args.dataset}/XC/raw_data/test.raw.txt')
            tst_cat = retain_topk(sp.load_npz(fname), k=5)
        else:
            tst_ids, tst_txt = load_raw_file(f'/data/datasets/{input_args.dataset}/XC/raw_data/test.raw.csv')
            tst_cat = retain_topk(sp.load_npz(fname), k=5)
        tst_cat = apply_threshold(tst_cat, input_args)

        tst_cat_txt = get_data_category(tst_cat, tst_txt, cat_txt)
        fname = (
            f'{output_dir}/raw_data/test_{meta_info_numbers[input_args.expt_no]}_{input_args.dataset}.raw.csv'
            if TYPE == 'prediction' else
            f'{output_dir}/raw_data/test_{meta_info_numbers[input_args.expt_no]}.raw.csv'
        )
        save_raw_file(fname, tst_ids, tst_cat_txt)

    elif input_args.type == "config":

        with open('configs/msmarco_data-ngame-category-linker.json') as file:
            early_fusion_config = json.load(file)
        
        config_file = f'/data/datasets/{input_args.dataset}/XC/configs/data.json'
        with open(config_file) as file:
            config = json.load(file)

        fname = f'{output_dir}/raw_data/test_{meta_info_numbers[input_args.expt_no]}_{input_args.dataset}.raw.csv'

        config_key = f"{input_args.dataset}_data-{meta_info_numbers[input_args.expt_no]}_{input_args.expt_no:03d}"

        early_fusion_config[config_key] = early_fusion_config.pop('msmarco_data-ngame-category-linker')
        early_fusion_config[config_key]['path'].pop('train', None)
        early_fusion_config[config_key]['path']['test'] = config['data']['path']['test']
        early_fusion_config[config_key]['path']['test']['data_info'] = fname
        early_fusion_config[config_key]['parameters']['main_max_lbl_sequence_length'] = 512
        
        with open(f'configs/beir/{config_key}.json', 'w') as file:
            json.dump(early_fusion_config, file, indent=4)

