import os, scipy.sparse as sp, numpy as np

from xclib.utils.sparse import retain_topk

if __name__ == "__main__":
    datasets = "arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"
    datasets = datasets.split(' ')

    data_dir = "/data/outputs/mogicX/"
    models = [
        "33_distilbert-beir-inference-001/predictions", 
        # "50_distilbert-ngame-category-linker-oracle-for-msmarco-002/predictions/47_msmarco-gpt-category-linker-002",
        # "56_bm25-for-beir-inference-001/predictions",
    ]

    for dataset in datasets:
        neg_mat, gt_mat = None, sp.load_npz(f'/data/datasets/beir/{dataset}/XC/trn_X_Y.npz')
        for model in models:
            fname = f"{data_dir}/{model}/train_predictions_{dataset.replace('/', '-')}.npz"
            mat = retain_topk(sp.load_npz(fname), k=50)
            assert gt_mat.shape == mat.shape
            neg_mat = mat if neg_mat is None else neg_mat + mat

        neg_mat = neg_mat.astype(np.int64)

        inter = neg_mat.multiply(gt_mat)
        r, c = inter.nonzero()
        neg_mat[r, c] = 0.0
        neg_mat.eliminate_zeros()

        sp.save_npz(f'/data/datasets/beir/{dataset}/XC/negtives_trn_X_Y.npz', neg_mat)

