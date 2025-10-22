import os, numpy as np

from tqdm.auto import tqdm

from sugar.core import *

if __name__ == '__main__':
    datasets = "msmarco arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"
    datasets = datasets.split(' ')

    data_dir = "/data/outputs/mogicX/47_msmarco-gpt-category-linker-002/raw_data/"
    for dataset in tqdm(datasets):
        dataset = dataset.replace("/", "-")

        cat_file = f'{data_dir}/test_gpt-category-ngame-linker_conflated_{dataset}.raw.csv'
        qry_ids, cat_txt = load_raw_file(cat_file)

        ent_file = f'{data_dir}/test_gpt-category-ngame-linker-conflated-wiki-entity_{dataset}.raw.csv'
        q_ids, ent_txt = load_raw_file(ent_file)

        assert np.all([x == y for x,y in zip(qry_ids, q_ids)])

        meta_txt = list()
        for c,e in zip(cat_txt, ent_txt):
            query, cat = c.split('<CATEGORIES>')
            q, ent = e.split('<CATEGORIES>')
            assert query == q
            meta_txt.append(query + '<CATEGORIES>' + cat + ' || ' + ent)

        save_file = f'{data_dir}/test_gpt-category-ngame-linker-conflated-wiki-entity-combined_{dataset}.raw.csv'
        save_raw_file(save_file, qry_ids, meta_txt)

