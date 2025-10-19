import multiprocessing as mp, numpy as np, scipy.sparse as sp, os

from rank_bm25 import BM25Okapi

from xcai.main import *
from sugar.core import *
from xclib.evaluation.xc_metrics import ndcg


def bm25_rank_chunk(args):
    queries, corpus, n = args
    bm25 = BM25Okapi(corpus, k1=0.9, b=0.4)

    data, indices = list(), list()
    for q in queries:
        scores = bm25.get_scores(q)
        top_n_idx = np.argsort(scores)[::-1][:n]
        data.append(scores[top_n_idx])
        indices.append(top_n_idx)

    return data, indices 


def bm25_inference(corpus, queries, top_n=200, num_processes=8):
    chunk_size = int(np.ceil(len(queries) / num_processes))
    query_chunks = [queries[i:i+chunk_size] for i in range(0, len(queries), chunk_size)]

    args = [(chunk, corpus, top_n) for chunk in query_chunks]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(bm25_rank_chunk, args)

    data, indices, indptr = [], [], [0]
    for sc, idx in results:
        for d,i in zip(sc, idx):
            data.extend(list(d))
            indices.extend(list(i))
            indptr.append(len(data))

    return sp.csr_matrix((data, indices, indptr), shape=(len(queries), len(corpus)), dtype=np.float32) 


if __name__ == "__main__":
    from rank_bm25 import BM25Okapi
    from nltk.tokenize import word_tokenize

    input_args = parse_args()

    raw_dir = f'/data/datasets/beir/{input_args.dataset}/XC/raw_data/'

    # load corpus
    fname = f'{raw_dir}/label.raw.txt' if input_args.dataset == "msmarco" else f'{raw_dir}/label.raw.csv'
    lbl_ids, corpus = load_raw_file(fname)

    # load queries
    fname = f'{raw_dir}/test.raw.txt' if input_args.dataset == "msmarco" else f'{raw_dir}/test.raw.csv'
    tst_ids, tst_txt = load_raw_file(fname)

    fname = f'{raw_dir}/train.raw.txt' if input_args.dataset == "msmarco" else f'{raw_dir}/train.raw.csv'
    trn_ids, trn_txt = load_raw_file(fname)

    tst_mat = sp.load_npz(f'/data/datasets/beir/{input_args.dataset}/XC/tst_X_Y.npz')
    tst_mat.data[:] = 1.0

    # Tokenize
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    tokenized_trn = [word_tokenize(q.lower()) for q in trn_txt]
    tokenized_tst = [word_tokenize(q.lower()) for q in tst_txt]

    # Run BM25 inference
    trn_pred = bm25_inference(tokenized_corpus, tokenized_trn, top_n=200, num_processes=10)
    tst_pred = bm25_inference(tokenized_corpus, tokenized_tst, top_n=200, num_processes=10)

    # save predictions
    output_dir = "/home/aiscuser/scratch1/outputs/mogicX/56_bm25-for-beir-inference-001/predictions"
    os.makedirs(output_dir, exist_ok=True)

    sp.save_npz(f'{output_dir}/test_predictions_{input_args.dataset.replace("/", "-")}.npz', tst_pred)
    sp.save_npz(f'{output_dir}/train_predictions_{input_args.dataset.replace("/", "-")}.npz', trn_pred)

    metric = ndcg(tst_pred, tst_mat, k=10)[-1] * 100
    print(f"{input_args.dataset}: {metric}")


