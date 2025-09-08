from sugar.core import *

import scipy.sparse as sp, os, argparse, json

from typing import List
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

from xcai.main import *

def get_data_label(data_lbl:sp.csr_matrix, data_txt:List, lbl_txt:List):
    assert data_lbl.shape[0] == len(data_txt), f'{data_lbl.shape[0]} != {len(data_txt)}'
    assert data_lbl.shape[1] == len(lbl_txt), f'{data_lbl.shape[1]} != {len(lbl_txt)}'

    data_lbl_txt = []
    for data, txt in tqdm(zip(data_lbl, data_txt), total=len(data_txt)):
        lbls = [lbl_txt[i] for i in data.indices]
        # data_lbl_txt.append(txt + " <DOCUMENTS> " + " || ".join(lbls))
        data_lbl_txt.append(" || ".join(lbls))
    return data_lbl_txt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    input_args = parse_args()

    output_dir = 'outputs/'
    os.makedirs(f'{output_dir}/raw_data', exist_ok=True)

    lbl_ids, lbl_txt = load_raw_file('/data/datasets/msmarco/XC/raw_data/label_exact.raw.txt')
    
    tst_ids, tst_txt = load_raw_file('/data/datasets/msmarco/XC/raw_data/test.raw.txt')
    tst_lbl = sp.load_npz('/data/datasets/msmarco/XC/tst_X_Y_exact.npz')

    tst_lbl_txt = get_data_label(tst_lbl, tst_txt, lbl_txt)

    # fname = f'{output_dir}/raw_data/test_label_msmarco.raw.csv'
    fname = f'{output_dir}/raw_data/label_label_msmarco.raw.csv'
    save_raw_file(fname, tst_ids, tst_lbl_txt)

