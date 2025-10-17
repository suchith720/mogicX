import scipy.sparse as sp, numpy as np
from tqdm.auto import tqdm

from xcai.main import *

import xclib.evaluation.xc_metrics as xm


def _ndcg(eval_flags, n, k=5):
    _cumsum = 0
    _dcg = np.cumsum(np.multiply(
        eval_flags, 1/np.log2(np.arange(k)+2)),
        axis=-1)
    ndcg = []
    for _k in range(k):
        _cumsum += 1/np.log2(_k+1+1)
        ndcg.append(np.multiply(_dcg[:, _k].reshape(-1, 1), 1/np.minimum(n, _cumsum)))
    return np.hstack(ndcg)


def ndcg(X, true_labels, k=5, sorted=False, use_cython=False):
    indices, true_labels, _, _ = xm._setup_metric(
        X, true_labels, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = xm._eval_flags(indices, true_labels, None)
    _total_pos = np.asarray(
        true_labels.sum(axis=1),
        dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1/np.log2(np.arange(1, _max_pos+1)+1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k)


if __name__ == "__main__":
    input_args = parse_args()

    pred_dir_1 = "/data/outputs/mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-002/predictions/"
    pred_dir_2 = "/data/outputs/mogicX/33_distilbert-beir-inference-001/predictions/"

    datasets="msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"
    datasets = datasets.split(' ')

    metric_1, metric_2, metric_3 = dict(), dict(), dict()
    for dataset in tqdm(datasets, total=len(datasets)):
        data_lbl_file = f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz"
        data_lbl = sp.load_npz(data_lbl_file)
        data_lbl.data[:] = 1.0

        pred_file_1 = f"{pred_dir_1}/test_predictions_{dataset.replace('/', '-')}.npz"
        pred_lbl_1 = sp.load_npz(pred_file_1)

        pred_file_2 = f"{pred_dir_2}/test_predictions_{dataset.replace('/', '-')}.npz"
        pred_lbl_2 = sp.load_npz(pred_file_2)

        m1 = ndcg(pred_lbl_1, data_lbl, k=10)[:, -1]
        m2 = ndcg(pred_lbl_2, data_lbl, k=10)[:, -1]

        metric_1[dataset] = np.mean(m1)
        metric_2[dataset] = np.mean(m2)
        metric_3[dataset] = np.mean(np.maximum(m1, m2))

    combine_datasets = "cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"
    combine_datasets = combine_datasets.split(' ')

    metric_1['cqadupstack'] = np.mean([metric_1.pop(dataset) for dataset in combine_datasets])
    metric_2['cqadupstack'] = np.mean([metric_2.pop(dataset) for dataset in combine_datasets])
    metric_3['cqadupstack'] = np.mean([metric_3.pop(dataset) for dataset in combine_datasets])

    print('50_distilbert-ngame-category-linker-oracle-for-msmarco-002')
    print(metric_1)
    print(f"Average: {np.mean(list(metric_1.values()))}")
    print()

    print('33_distilbert-beir-inference-001')
    print(metric_2)
    print(f"Average: {np.mean(list(metric_2.values()))}")
    print()

    print('Maximum')
    print(metric_3)
    print(f"Average: {np.mean(list(metric_3.values()))}")


