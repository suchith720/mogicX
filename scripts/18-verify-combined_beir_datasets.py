import scipy.sparse as sp, os, numpy as np

from tqdm.auto import tqdm

from sugar.core import *

def load_data(dataset):
    trn_mat_file = f"/data/datasets/beir/{dataset}/XC/trn_X_Y.npz"
    trn_mat = sp.load_npz(trn_mat_file)

    tst_mat_file = f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz"
    tst_mat = sp.load_npz(tst_mat_file)

    return trn_mat, tst_mat

if __name__ == "__main__":
    datasets = "msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"
    datasets = datasets.split(" ")

    trn_lbl = sp.load_npz('/data/datasets/beir/all-beir/XC/trn_X_Y.npz')
    tst_lbl = sp.load_npz('/data/datasets/beir/all-beir/XC/tst_X_Y.npz')

    total_trn_r, total_tst_r, total_l = 0, 0, 0
    for dataset in tqdm(datasets):
        trn_mat, tst_mat = load_data(dataset)
        assert trn_mat.shape[1] == tst_mat.shape[1]

        def check(data_lbl, mat, total_r):
            r, l = mat.shape

            m = data_lbl[total_r:total_r+r].tocsr()

            assert m[:,:total_l].nnz == 0
            assert m[:,total_l+l:].nnz == 0

            m = m[:,total_l:total_l+l].tocsr()
            assert (m != mat).nnz == 0

        check(trn_lbl, trn_mat, total_trn_r)
        check(tst_lbl, tst_mat, total_tst_r)

        total_trn_r += trn_mat.shape[0]
        total_tst_r += tst_mat.shape[0]
        total_l += trn_mat.shape[1]


