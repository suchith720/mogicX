import scipy.sparse as sp, os

from tqdm.auto import tqdm

from sugar.core import *

def load_data(dataset):
    trn_raw_file = (
        f"/data/datasets/beir/{dataset}/XC/raw_data/train.raw.txt"
        if dataset == "msmarco" else
        f"/data/datasets/beir/{dataset}/XC/raw_data/train.raw.csv"
    )
    trn_mat_file = f"/data/datasets/beir/{dataset}/XC/trn_X_Y.npz"
    trn_ids, trn_txt = load_raw_file(trn_raw_file)
    trn_mat = sp.load_npz(trn_mat_file)

    tst_raw_file = (
        f"/data/datasets/beir/{dataset}/XC/raw_data/test.raw.txt"
        if dataset == "msmarco" else
        f"/data/datasets/beir/{dataset}/XC/raw_data/test.raw.csv"
    )
    tst_mat_file = f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz"
    tst_ids, tst_txt = load_raw_file(tst_raw_file)
    tst_mat = sp.load_npz(tst_mat_file)

    lbl_raw_file = (
        f"/data/datasets/beir/{dataset}/XC/raw_data/label.raw.txt"
        if dataset == "msmarco" else
        f"/data/datasets/beir/{dataset}/XC/raw_data/label.raw.csv"
    )
    lbl_ids, lbl_txt = load_raw_file(lbl_raw_file)

    return (trn_ids, trn_txt, trn_mat), (tst_ids, tst_txt, tst_mat), (lbl_ids, lbl_txt)


def save_data(trn_info, tst_info, lbl_info):
    data_dir = '/data/datasets/beir/all-beir/XC'
    os.makedirs(data_dir, exist_ok=True)

    raw_dir = '/data/datasets/beir/all-beir/XC/raw_data'
    os.makedirs(raw_dir, exist_ok=True)

    # train
    trn_raw_file = f"{raw_dir}/train.raw.csv"
    save_raw_file(trn_raw_file, trn_info[0], trn_info[1])

    trn_mat_file = f"{data_dir}/trn_X_Y.npz"
    sp.save_npz(trn_mat_file, trn_info[2])

    # test
    tst_raw_file = f"{raw_dir}/test.raw.csv"
    save_raw_file(tst_raw_file, tst_info[0], tst_info[1])

    tst_mat_file = f"{data_dir}/tst_X_Y.npz"
    sp.save_npz(tst_mat_file, tst_info[2])

    # label
    lbl_raw_file = f"{raw_dir}/label.raw.csv"
    save_raw_file(lbl_raw_file, lbl_info[0], lbl_info[1])



if __name__ == "__main__":
    datasets = "msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"
    datasets = datasets.split(" ")

    trn_ids, trn_txt = [], []
    tst_ids, tst_txt = [], []
    lbl_ids, lbl_txt = [], []

    trn_mat, tst_mat = [], []

    for dataset in tqdm(datasets):
        trn, tst, lbl = load_data(dataset)

        lbl_ids.extend(lbl[0])
        lbl_txt.extend(lbl[1])

        trn_ids.extend(trn[0])
        trn_txt.extend(trn[1])

        tst_ids.extend(tst[0])
        tst_txt.extend(tst[1])

        trn_mat.append(trn[2])
        tst_mat.append(tst[2])

    def combine_mat(mats):
        num_lbls, resz_mats = 0, []
        for m in mats:
            resz_mat = sp.csr_matrix((m.data, m.indices + num_lbls, m.indptr), shape=(m.shape[0], len(lbl_ids)))
            num_lbls += m.shape[1]
            resz_mats.append(resz_mat)
        return sp.vstack(resz_mats)

    trn_mat = combine_mat(trn_mat)
    tst_mat = combine_mat(tst_mat)

    assert len(trn_ids) == trn_mat.shape[0]
    assert len(tst_ids) == tst_mat.shape[0]
    assert len(lbl_ids) == trn_mat.shape[1] == tst_mat.shape[1]

    save_data((trn_ids, trn_txt, trn_mat), (tst_ids, tst_txt, tst_mat), (lbl_ids, lbl_txt))



