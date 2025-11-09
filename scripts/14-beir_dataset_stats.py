import scipy.sparse as sp, os

if __name__ == '__main__':
    
    datasets="msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq nq-train quora scidocs scifact webis-touche2020 trec-covid cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

    total_trn, total_lbls = 0, 0

    for dset in datasets.split(' '):
        print(f'{dset}:')
        trn_file = f"/data/datasets/beir/{dset}/XC/trn_X_Y.npz"

        if os.path.exists(trn_file):
            trn_lbl = sp.load_npz(trn_file)
            print(f'Train shape: {trn_lbl.shape[0]} x {trn_lbl.shape[1]}')

            total_trn += trn_lbl.shape[0]
            total_lbls += trn_lbl.shape[1]

        tst_file = f"/data/datasets/beir/{dset}/XC/tst_X_Y.npz"
        if os.path.exists(tst_file):
            tst_lbl = sp.load_npz(tst_file)
            print(f'Test shape: {tst_lbl.shape[0]} x {tst_lbl.shape[1]}')

        print()

    print(f'Total training points: {total_trn}')
    print(f'Total labels: {total_lbls}')


