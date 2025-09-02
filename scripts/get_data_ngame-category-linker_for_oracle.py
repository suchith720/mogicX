from sugar.core import *

import scipy.sparse as sp, os

from typing import List
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

def get_data_category(data_cat:sp.csr_matrix, data_txt:List, cat_txt:List):
    assert data_cat.shape[0] == len(data_txt), f'{data_cat.shape[0]} != {len(data_txt)}'
    assert data_cat.shape[1] == len(cat_txt), f'{data_cat.shape[1]} != {len(cat_txt)}'

    data_cat_txt = []
    for data, txt in tqdm(zip(data_cat, data_txt), total=len(data_txt)):
        cats = [cat_txt[i] for i in data.indices]
        data_cat_txt.append(txt + " <CATEGORIES> " + " || ".join(cats))
    return data_cat_txt


if __name__ == '__main__':
    output_dir = '/home/aiscuser/scratch1/outputs/mogicX/47_msmarco-gpt-category-linker-001'
    os.makedirs(f'{output_dir}/raw_data', exist_ok=True)

    cat_ids, cat_txt = load_raw_file('/data/datasets/msmarco/XC/raw_data/category-gpt.raw.csv')
    
    trn_ids, trn_txt = load_raw_file('/data/datasets/msmarco/XC/raw_data/train.raw.txt')
    trn_cat = retain_topk(sp.load_npz(f'{output_dir}/predictions/train_predictions.npz'), k=5)

    trn_cat_txt = get_data_category(trn_cat, trn_txt, cat_txt)
    save_raw_file(f'{output_dir}/raw_data/train_ngame-gpt-category-linker.raw.csv', trn_ids, trn_cat_txt)
    
    tst_ids, tst_txt = load_raw_file('/data/datasets/msmarco/XC/raw_data/test.raw.txt')
    tst_cat = retain_topk(sp.load_npz(f'{output_dir}/predictions/test_predictions.npz'), k=5)

    tst_cat_txt = get_data_category(tst_cat, tst_txt, cat_txt)
    save_raw_file(f'{output_dir}/raw_data/test_ngame-gpt-category-linker.raw.csv', tst_ids, tst_cat_txt)

