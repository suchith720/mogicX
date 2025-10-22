import torch, scipy.sparse as sp, os

from xcai.main import *
from xcai.data import BaseXCDataset

from sugar.core import load_raw_file

if __name__ == '__main__':
    input_args = parse_args()

    data_dir = f"/data/datasets/beir/{input_args.dataset}/XC/"

    output_dir = f"/data/outputs/mogicX/54_nvembed-for-msmarco-001/predictions/{input_args.dataset}"

    if input_args.dataset == "msmarco":
        print("Loading data ...")

        trn_repr = torch.load(f"{output_dir}/trn_repr.pth")

        lbl_repr = torch.load(f"{output_dir}/lbl_repr.pth")
        lbl_ids, lbl_txt = load_raw_file(f"{data_dir}/raw_data/label.raw.txt")
        lbl_ids2idx = {k:i for i,k in enumerate(lbl_ids)}

        neg_ids, neg_txt = load_raw_file(f"{data_dir}/raw_data/ce-scores.raw.txt")
        neg_idx = [lbl_ids2idx[ids] for ids in neg_ids]
        neg_repr = lbl_repr[neg_idx]

        lbl_exact_ids, lbl_exact_txt = load_raw_file(f"{data_dir}/raw_data/label_exact.raw.txt")
        lbl_exact_idx = [lbl_ids2idx[ids] for ids in lbl_exact_ids]
        lbl_repr = lbl_repr[lbl_exact_idx]

        trn_mat = sp.load_npz(f"{data_dir}/trn_X_Y_exact.npz")
        neg_mat = sp.load_npz(f"{data_dir}/ce-negatives-topk-05_trn_X_Y.npz")

    else:
        raise ValueError(f"Invalid dataset: {input_args.dataset}")

    print("Scoring matrix ...")

    save_dir = f"/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/{input_args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    assert trn_mat.shape == (trn_repr.shape[0], lbl_repr.shape[0])
    curr_trn_mat = BaseXCDataset.score_data_lbl(trn_mat, trn_repr.float(), lbl_repr.float(), batch_size=1024, normalize=input_args.normalize)
    sp.save_npz(f"{save_dir}/trn_X_Y_normalize.npz" if input_args.normalize else f"{save_dir}/trn_X_Y.npz", curr_trn_mat)

    assert neg_mat.shape == (trn_repr.shape[0], neg_repr.shape[0])
    curr_neg_mat = BaseXCDataset.score_data_lbl(neg_mat, trn_repr.float(), neg_repr.float(), batch_size=1024, normalize=input_args.normalize)
    sp.save_npz(f"{save_dir}/negatives_trn_X_Y_normalize.npz" if input_args.normalize else f"{save_dir}/negatives_trn_X_Y.npz", curr_neg_mat)

