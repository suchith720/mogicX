import os

from xcai.main import *
from sugar.core import load_raw_file, save_raw_file

if __name__ == '__main__':
    input_args = parse_args()

    fname = f'/data/datasets/{input_args.dataset}/XC/raw_data/test.raw.csv'
    tst_ids, tst_txt = load_raw_file(fname)

    output_dir = f'/home/aiscuser/datasets/{input_args.dataset}' 
    os.makedirs(output_dir, exist_ok=True)

    save_raw_file(f'{output_dir}/test.raw.txt', tst_ids, tst_txt, sep='\t')

