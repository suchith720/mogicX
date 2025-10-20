import scipy.sparse as sp, os, argparse, json

from typing import List, Dict, Optional
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

from xcai.main import *
from xcai.config import PARAM

from sugar.core import *


def get_data_metadata(data_meta:sp.csr_matrix, data_txt:List, meta_txt:List, meta_type:Optional[str]="<CATEGORIES>"):
    assert data_meta.shape[0] == len(data_txt), f'{data_meta.shape[0]} != {len(data_txt)}'
    assert data_meta.shape[1] == len(meta_txt), f'{data_meta.shape[1]} != {len(meta_txt)}'

    data_meta_txt = []
    for data, txt in tqdm(zip(data_meta, data_txt), total=len(data_txt)):
        metas = [meta_txt[i] for i in data.indices]
        data_meta_txt.append(txt + f" {meta_type} " + " || ".join(metas))
    return data_meta_txt


def apply_threshold(mat:sp.csr_matrix, abs_thresh:Optional[int]=None, diff_thresh:Optional[int]=None):
    if abs_thresh is not None: 
        mat = Filter.threshold(mat, t=abs_thresh)
    if diff_thresh is not None: 
        mat = Filter.difference(mat, t=diff_thresh)
    return mat


def get_combined_raw_file(output_dir:str, type:str, dataset:str, expt_no:int, meta_info:Dict, output_info:Dict, save_dir_name:Optional[str]=None, 
                          abs_thresh:Optional[int]=None, diff_thresh:Optional[int]=None, save_train_raw:Optional[bool]=False):
    meta_ids, meta_txt = load_raw_file(f'/data/datasets/beir/msmarco/XC/raw_data/{meta_info[expt_no]}.raw.csv')
    save_dir_name = "predictions" if save_dir_name is None else save_dir_name

    # Train dataset
    if save_train_raw: 
        fname = (
            (
                f'{output_dir}/{save_dir_name}/train_predictions.npz' 
                if dataset == "msmarco" else 
                f'{output_dir}/{save_dir_name}/train_predictions_{dataset.replace("/","-")}.npz'
            )
            if type == 'prediction' else 
            f'{output_dir}/{meta_info[expt_no]}_trn_X_Y.npz'
        )
        if dataset == "msmarco":
            trn_ids, trn_txt = load_raw_file(f'/data/datasets/beir/{dataset}/XC/raw_data/train.raw.txt')
        else:
            trn_ids, trn_txt = load_raw_file(f'/data/datasets/beir/{dataset}/XC/raw_data/train.raw.csv')

        if os.path.exists(fname):
            trn_meta = retain_topk(sp.load_npz(fname), k=5)
            trn_meta = apply_threshold(trn_meta, abs_thresh=abs_thresh, diff_thresh=diff_thresh)

            trn_meta_txt = get_data_metadata(trn_meta, trn_txt, meta_txt)
            fname = (
                f'{output_dir}/raw_data/train_{output_info[expt_no]}_{dataset.replace("/", "-")}.raw.csv'
                if type == 'prediction' else
                f'{output_dir}/raw_data/train_{output_info[expt_no]}.raw.csv'
            )
            save_raw_file(fname, trn_ids, trn_meta_txt)

    # Test dataset
    fname = (
        (
            f'{output_dir}/{save_dir_name}/test_predictions.npz' 
            if dataset == "msmarco" else 
            f'{output_dir}/{save_dir_name}/test_predictions_{dataset.replace("/","-")}.npz'
        )
        if type == 'prediction' else 
        f'{output_dir}/{meta_info[expt_no]}_tst_X_Y.npz'
    )
    if dataset == "msmarco":
        tst_ids, tst_txt = load_raw_file(f'/data/datasets/beir/{dataset}/XC/raw_data/test.raw.txt')
    else:
        tst_ids, tst_txt = load_raw_file(f'/data/datasets/beir/{dataset}/XC/raw_data/test.raw.csv')
    tst_meta = retain_topk(sp.load_npz(fname), k=5)
    tst_meta = apply_threshold(tst_meta, abs_thresh=abs_thresh, diff_thresh=diff_thresh)

    tst_meta_txt = get_data_metadata(tst_meta, tst_txt, meta_txt)
    fname = (
        f'{output_dir}/raw_data/test_{output_info[expt_no]}_{dataset.replace("/", "-")}.raw.csv'
        if type == 'prediction' else
        f'{output_dir}/raw_data/test_{output_info[expt_no]}.raw.csv'
    )
    save_raw_file(fname, tst_ids, tst_meta_txt)


def get_config_file(output_dir:str, dataset:str, expt_no:int, output_info:Dict):
    config_file = f'/data/datasets/beir/{dataset}/XC/configs/data.json'
    with open(config_file) as file:
        config = json.load(file)

    config_key = f"{dataset.replace('/', '-')}_data-{output_info[expt_no]}"
    config[config_key] = config.pop('data')

    # train config
    fname = f'{output_dir}/raw_data/train_{output_info[expt_no]}_{dataset.replace("/", "-")}.raw.csv'
    if os.path.exists(fname): 
        config[config_key]["path"]["train"]["data_info"] = fname
    else:
        config[config_key]["path"]["test"]["lbl_info"] = config[config_key]["path"]["train"]["lbl_info"]
        del config[config_key]["path"]["train"]

    # test config
    fname = f'{output_dir}/raw_data/test_{output_info[expt_no]}_{dataset.replace("/", "-")}.raw.csv'
    config[config_key]["path"]["test"]["data_info"] = fname

    config[config_key]['parameters'] = PARAM
    config[config_key]['parameters']['main_max_data_sequence_length'] = 300
    config[config_key]['parameters']['main_max_lbl_sequence_length'] = 512
    
    with open(f'configs/beir/{input_args.dataset}/{config_key}.json', 'w') as file:
        json.dump(config, file, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="raw")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--expt_no', type=int, required=True)
    parser.add_argument('--type', type=str, default="prediction")
    parser.add_argument('--save_dir_name', type=str, default=None)
    parser.add_argument('--abs_thresh', type=float, default=None)
    parser.add_argument('--diff_thresh', type=float, default=None)
    parser.add_argument('--save_train_info', action='store_true') 
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    input_args = parse_args()

    META_INFO = {
        1: f'category-gpt', 
        2: f'category-gpt_conflated', 
        # 2: f'wiki-entity_ngame', 

        7: f'category-gpt-linker_conflated-001_conflated-001', 
        8: f'category-gpt-linker_conflated-001_conflated-001', 
        9: f'category-gpt-linker_conflated-001_conflated-001', 
    }

    OUTPUT_INFO = {
        1: f'gpt-category-ngame-linker', 
        2: f'gpt-category-ngame-linker_conflated',
        # 2: f'gpt-category-ngame-linker-conflated-wiki-entity',

        7: f'gpt-category-linker-ngame-linker_conflated-001-conflated-001-007',
        8: f'gpt-category-linker-ngame-linker_conflated-001-conflated-001-008',
        9: f'gpt-category-linker-ngame-linker_conflated-001-conflated-001-009',
    }

    # Output directory
    output_dir = (
        f'/data/outputs/mogicX/47_msmarco-gpt-category-linker-{input_args.expt_no:03d}' 
        if input_args.type == 'prediction' else 
        f'/data/datasets/beir/{input_args.dataset}/XC/'
    )
    os.makedirs(f'{output_dir}/raw_data', exist_ok=True)


    if input_args.task == "raw": 
        get_combined_raw_file(output_dir, input_args.type, input_args.dataset, expt_no=input_args.expt_no, meta_info=META_INFO, output_info=OUTPUT_INFO,
                              save_dir_name=input_args.save_dir_name, abs_thresh=input_args.abs_thresh, diff_thresh=input_args.diff_thresh, 
                              save_train_raw=input_args.save_train_info)
    elif input_args.task == "config":
        get_config_file(output_dir, input_args.dataset, expt_no=input_args.expt_no, output_info=OUTPUT_INFO)


