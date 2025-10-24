import multiprocessing as mp, numpy as np, scipy.sparse as sp, os, bm25s, Stemmer, argparse

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from typing import Optional, List
from tqdm.auto import tqdm

from xcai.main import *
from sugar.core import *
from xclib.evaluation.xc_metrics import ndcg

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

h = logging.StreamHandler()
h.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
h.setFormatter(formatter)
logger.addHandler(h)

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


def rank_bm25_inference(corpus:List, trn_txt:List, tst_txt:List, top_n:Optional[int]=200, num_processes:Optional[int]=10):
    # Tokenize
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    tokenized_trn = [word_tokenize(q.lower()) for q in trn_txt]
    tokenized_tst = [word_tokenize(q.lower()) for q in tst_txt]

    # Run BM25 inference
    trn_pred = bm25_inference(tokenized_corpus, tokenized_trn, top_n=top_n, num_processes=num_processes)
    tst_pred = bm25_inference(tokenized_corpus, tokenized_tst, top_n=top_n, num_processes=num_processes)

    return trn_pred, tst_pred


def bm25s_chunk(args):
    queries, corpus, k = args

    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer, stopwords="en")
    corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple")

    retriever = bm25s.BM25(backend="numba")
    retriever.index(corpus_tokens)

    toks = tokenizer.tokenize(queries)
    indices, scores = retriever.retrieve(toks, k=k)
    indptr = np.hstack([np.zeros(1, dtype=np.int64), np.full((len(queries),), k).cumsum()])
    pred = sp.csr_matrix((scores.flatten(), indices.flatten(), indptr), shape=(len(queries), len(corpus)), dtype=np.float32)

    return pred


def _bm25s_inference(corpus:List, queries:List, k:Optional[int]=10, num_processes:Optional[int]=10):
    chunk_size = int(np.ceil(len(queries) / num_processes))
    query_chunks = [queries[i:i+chunk_size] for i in range(0, len(queries), chunk_size)]

    args = [(chunk, corpus, k) for chunk in query_chunks]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(bm25s_chunk, args)

    pred = sp.vstack([pred for pred in results])
    assert pred.shape[0] == len(queries)
    assert pred.shape[1] == len(corpus)
    return pred


def bm25s_inference(corpus:List, trn_txt:List, tst_txt:List, k:Optional[int]=10, num_processes:Optional[int]=10):
    logger.info('Computing train-set predictions ...')
    trn_pred = _bm25s_inference(corpus, trn_txt, k, num_processes)

    logger.info('Computing test-set predictions ...')
    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer, stopwords="en")
    corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple")

    retriever = bm25s.BM25(backend="numba")
    retriever.index(corpus_tokens)

    toks = tokenizer.tokenize(tst_txt)
    indices, scores = retriever.retrieve(toks, k=k)
    indptr = np.hstack([np.zeros(1, dtype=np.int64), np.full((len(tst_txt),), k).cumsum()])
    tst_pred = sp.csr_matrix((scores.flatten(), indices.flatten(), indptr), shape=(len(tst_txt), len(corpus)))

    return trn_pred, tst_pred


def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='bm25s')
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    input_args = parse_args()
    extra_args = additional_args()

    raw_dir = f'/data/datasets/beir/{input_args.dataset}/XC/raw_data/'

    logger.info('Loading data ...')
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
    
    if extra_args.type == "rank_bm25":
        trn_pred, tst_pred = rank_bm25_inference(corpus, trn_txt, tst_txt, top_n=10, num_processes=10)
    elif extra_args.type == "bm25s":
        trn_pred, tst_pred = bm25s_inference(corpus, trn_txt, tst_txt, k=10, num_processes=10)
    else:
        raise ValueError(f'Invalid type: {extra_args.type}')

    # save predictions
    output_dir = "/home/aiscuser/scratch1/outputs/mogicX/56_bm25-for-beir-inference-001/predictions"
    os.makedirs(output_dir, exist_ok=True)

    sp.save_npz(f'{output_dir}/test_predictions_{input_args.dataset.replace("/", "-")}.npz', tst_pred)
    sp.save_npz(f'{output_dir}/train_predictions_{input_args.dataset.replace("/", "-")}.npz', trn_pred)

    metric = ndcg(tst_pred, tst_mat, k=10)[-1] * 100
    print(f"{input_args.dataset}: {metric}")


