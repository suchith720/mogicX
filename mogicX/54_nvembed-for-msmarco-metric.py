import os, torch, json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse, math, torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from xcai.metrics import *

def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_parts', type=int)
    parser.add_argument('--combine_lbl_embed', action='store_true')
    parser.add_argument('--combine_trn_embed', action='store_true')
    parser.add_argument('--combine_tst_embed', action='store_true')

    parser.add_argument('--compute_metrics', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    output_dir = '/home/aiscuser/scratch1/outputs/mogicX/54_nvembed-for-msmarco-001/predictions'
    extra_args = additional_args()

    if extra_args.combine_lbl_embed:
        lbl_repr = torch.vstack([torch.load(f'{output_dir}/lbl_repr_{idx:03d}.pth') for idx in range(extra_args.num_parts)])
        torch.save(lbl_repr, f'{output_dir}/lbl_repr.pth')

    if extra_args.combine_trn_embed:
        trn_repr = torch.vstack([torch.load(f'{output_dir}/trn_repr_{idx:03d}.pth') for idx in range(extra_args.num_parts)])
        torch.save(trn_repr, f'{output_dir}/trn_repr.pth')

    if extra_args.combine_tst_embed:
        tst_repr = torch.vstack([torch.load(f'{output_dir}/tst_repr_{idx:03d}.pth') for idx in range(extra_args.num_parts)])
        torch.save(tst_repr, f'{output_dir}/tst_repr.pth')

    if extra_args.compute_metrics:
        tst_mat = sp.load_npz('/data/datasets/beir/msmarco/XC/tst_X_Y.npz')

        tst_repr = torch.load(f'{output_dir}/tst_repr.pth')
        lbl_repr = torch.load(f'{output_dir}/lbl_repr.pth')

        tst_repr = F.normalize(tst_repr, dim=1) if extra_args.normalize else tst_repr
        lbl_repr = F.normalize(lbl_repr, dim=1) if extra_args.normalize else lbl_repr
        lbl_repr = lbl_repr.T

        metric = PrecReclMrr(lbl_repr.shape[1], pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])
        metric.reset()

        data_batch_sz, lbl_batch_sz = 1000, 10_000
        for i in tqdm(range(0, tst_repr.shape[0], data_batch_sz)):
            rep, gt = tst_repr[i:i+data_batch_sz].to('cuda'), tst_mat[i:i+data_batch_sz]

            # Compute scores
            scores = []
            for j in tqdm(range(0, lbl_repr.shape[1], lbl_batch_sz)):
                sc = rep@lbl_repr[:, j:j+lbl_batch_sz].to('cuda')
                scores.append(sc.to('cpu'))
            scores = torch.hstack(scores)

            # Top-k scores
            scores, indices = torch.topk(scores, k=200, dim=1, largest=True)
            o = {
                'pred_score': scores.flatten().to(torch.float32),
                'pred_idx': indices.flatten().to(torch.float32),
                'pred_ptr': torch.full((len(rep),), 200, dtype=torch.int64),
                'targ_idx': torch.tensor(gt.indices, dtype=torch.int64),
                'targ_ptr': torch.tensor([p-q for p,q in zip(gt.indptr[1:], gt.indptr)], dtype=torch.int64),
            }
            metric.accumulate(**o)

        print(metric.value)



