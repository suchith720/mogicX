import scipy.sparse as sp, numpy as np
from tqdm.auto import tqdm

from xcai.main import *

import xclib.evaluation.xc_metrics as xm
from xclib.utils.sparse import retain_topk

DATASETS = ['msmarco', 'arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa', 'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs', 
        'scifact', 'webis-touche2020', 'trec-covid', 'cqadupstack/android', 'cqadupstack/english', 'cqadupstack/gaming', 'cqadupstack/gis', 
        'cqadupstack/mathematica', 'cqadupstack/physics', 'cqadupstack/programmers', 'cqadupstack/stats', 'cqadupstack/tex', 'cqadupstack/unix', 
        'cqadupstack/webmasters', 'cqadupstack/wordpress']

COMBINE_DATASETS = ['cqadupstack/android', 'cqadupstack/english', 'cqadupstack/gaming', 'cqadupstack/gis', 'cqadupstack/mathematica', 
        'cqadupstack/physics', 'cqadupstack/programmers', 'cqadupstack/stats', 'cqadupstack/tex', 'cqadupstack/unix', 'cqadupstack/webmasters', 
        'cqadupstack/wordpress']


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

    pred_dirs = [
        "/data/outputs/mogicX/50_distilbert-ngame-category-linker-oracle-for-msmarco-002/predictions/47_msmarco-gpt-category-linker-002/",
        "/data/outputs/mogicX/33_distilbert-beir-inference-001/predictions/",
    ]
    meta_dir = "/data/outputs/mogicX/47_msmarco-gpt-category-linker-002/predictions/"


    metrics, dset_len, info = {}, [], []
    corel = {i: dict() for i in range(1,4)}

    for dataset in tqdm(DATASETS, total=len(DATASETS)):
        data_lbl = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz")
        data_lbl.data[:] = 1.0

        score_mat = []
        for dirname in pred_dirs:
            pred_lbl = sp.load_npz(f"{dirname}/test_predictions_{dataset.replace('/', '-')}.npz")
            score_mat.append(ndcg(pred_lbl, data_lbl, k=10)[:, -1])
            m = metrics.setdefault(dirname.split('/')[4], {})
            m[dataset] = np.mean(score_mat[-1])

        m = metrics.setdefault('Maximum', {}) 
        m[dataset] = np.mean(np.maximum(score_mat[0], score_mat[1]))

        # score corelations
        fname = "test_predictions.npz" if dataset == "msmarco" else f"test_predictions_{dataset.replace('/', '-')}.npz"
        data_meta = sp.load_npz(f"{meta_dir}/{fname}")
        data_meta = retain_topk(data_meta, k=3)

        idx = score_mat[0] >= score_mat[1]
        scores = np.sort(data_meta.data.reshape(-1, 3), axis=1)
        info.append(np.hstack([score_mat[0].reshape(-1, 1), score_mat[1].reshape(-1, 1), idx.astype(np.float32).reshape(-1, 1), scores]))

        dset_len.append(len(idx))

        for i,sc in enumerate(np.mean(scores[idx], axis=0) - np.mean(scores[~idx], axis=0)):
            corel[3-i][dataset] = sc

    for v in metrics.values(): v['cqadupstack'] = np.mean([v.pop(dataset) for dataset in COMBINE_DATASETS])


    for k,v in metrics.items():
        print(k, ':', v)
        print(f"Average: {np.mean(list(v.values()))}")
        print()

    for k,v in corel.items():
        print(f'Top-{k}', ':', v)
        print()

    info = np.vstack(info)
    assert info.shape[0] == sum(dset_len)
    save_dir = '/home/aiscuser/scratch1/tmp/'
    np.save(f'{save_dir}/info.npy', info) 
    np.save(f'{save_dir}/dset_len.npy', np.array(dset_len))

