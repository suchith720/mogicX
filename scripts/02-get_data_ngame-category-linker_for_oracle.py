from sugar.core import *

import scipy.sparse as sp, os, argparse, json

from typing import List
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

from xcai.main import *

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
    parser.add_argument('--dataset', type=str)
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    input_args = parse_args()

    output_dir = '/data/outputs/mogicX/47_msmarco-gpt-category-linker-001'
    os.makedirs(f'{output_dir}/raw_data', exist_ok=True)

    cat_ids, cat_txt = load_raw_file('/data/datasets/msmarco/XC/raw_data/category-gpt.raw.csv')
    
    # trn_ids, trn_txt = load_raw_file(f'/data/datasets/{input_args.dataset}/XC/raw_data/train.raw.txt')
    # trn_cat = retain_topk(sp.load_npz(f'{output_dir}/predictions/train_predictions_{input_args.dataset}.npz'), k=5)

    # trn_cat_txt = get_data_category(trn_cat, trn_txt, cat_txt)
    # save_raw_file(f'{output_dir}/raw_data/train_ngame-gpt-category-linker_{input_args.dataset}.raw.csv', trn_ids, trn_cat_txt)
    
    tst_ids, tst_txt = load_raw_file(f'/data/datasets/{input_args.dataset}/XC/raw_data/test.raw.csv')
    tst_cat = retain_topk(sp.load_npz(f'{output_dir}/predictions/test_predictions_{input_args.dataset}.npz'), k=3)

    tst_cat = Filter.threshold(tst_cat, t=0.2)
    tst_cat = Filter.difference(tst_cat, t=0.1)

    tst_cat_txt = get_data_category(tst_cat, tst_txt, cat_txt)
    fname = f'{output_dir}/raw_data/test_ngame-gpt-category-linker_{input_args.dataset}.raw.csv'
    save_raw_file(fname, tst_ids, tst_cat_txt)

    # with open('configs/data-ngame-category-linker.json') as file:
    #     early_fusion_config = json.load(file)
    # 
    # config_file = f'/data/datasets/{input_args.dataset}/XC/configs/data.json'
    # with open(config_file) as file:
    #     config = json.load(file)

    # early_fusion_config['data-ngame-category-linker']['path'].pop('train', None)
    # early_fusion_config['data-ngame-category-linker']['path']['test'] = config['data']['path']['test']
    # early_fusion_config['data-ngame-category-linker']['path']['test']['data_info'] = fname

    # early_fusion_config['data-ngame-category-linker']['parameters']['main_max_lbl_sequence_length'] = 512
    # 
    # config_key = f'{input_args.dataset}-data-ngame-category-linker'
    # with open(f'configs/{config_key}.json', 'w') as file:
    #     json.dump({config_key: early_fusion_config['data-ngame-category-linker']}, file, indent=4)

