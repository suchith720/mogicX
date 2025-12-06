import os, numpy as np, torch, joblib

from xcai.clustering.cluster import *

if __name__ == "__main__":
    datasets = ['msmarco', 'climate-fever', 'fever', 'hotpotqa', 'dbpedia-entity', 'nq']

    data_dir = "/data/outputs/mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-008/predictions/"

    idx2dset = {i:dset for i,dset in enumerate(datasets)}

    # load data
    indptr, lbl_repr = [0], []
    for dset in datasets:
        file = f"{data_dir}/label_repr_{dset}-008-007.pth"
        lbl_repr.append(torch.load(file))
        indptr.append(indptr[-1] + len(lbl_repr[-1]))
    lbl_repr = torch.vstack(lbl_repr)

    # cluster documents
    clusters = BalancedClusters.proc(lbl_repr[:1000], min_cluster_sz=10)

    # sample indices
    indices = [np.random.choice(c) for c in clusters]

    samples, p = dict(), 0
    for i in indices:
        if i >= indptr[p+1]: p += 1
        samples.setdefault(p, []).append(i - indptr[p])

    # save indices
    dset2ind = {idx2dset[k]:v for k,v in samples.items()}
    joblib.dump(dset2ind, "/home/aiscuser/beir_sample_indices.joblib")

