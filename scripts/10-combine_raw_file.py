import scipy.sparse as sp, numpy as np, json, argparse

from sugar.core import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="raw")
    return parser.parse_known_args()[0]

if __name__ == '__main__':

    args = parse_args()
    TYPE = args.type

    datasets = "nq fiqa hotpotqa fever dbpedia quora trec-covid climate-fever scifact scidocs arguana nfcorpus"
    for dataset in datasets.split(' '):
        print(dataset)

        if TYPE == "raw":
            file_1 = f'/data/share/from_deepak/datasets/{dataset}/raw_data/test_gpt-category-linker.raw.csv'
            file_2 = f'/data/outputs/mogicX/47_msmarco-gpt-category-linker-001/raw_data/test_ngame-gpt-category-linker_{dataset}.raw.csv'

            ids_1, txt_1 = load_raw_file(file_1)
            ids_2, txt_2 = load_raw_file(file_2)

            assert np.all([p == q for p,q in zip(ids_1, ids_2)])
            comb_txt = [p + q.split('<CATEGORIES>')[1] for p,q in zip(txt_1, txt_2)]
            
            save_file = f'/data/share/from_deepak/datasets/{dataset}/raw_data/test_ngame-gpt-category-linker-combined.raw.csv'
            save_raw_file(save_file, ids_1, comb_txt)

        elif TYPE == "config":
            config_file = f'configs/beir/{dataset}_data-ngame-category-linker.json'
            with open(config_file) as file:
                config = json.load(file)

            save_file = f'/data/share/from_deepak/datasets/{dataset}/raw_data/test_ngame-gpt-category-linker-combined.raw.csv'

            config[f'{dataset}_data-ngame-category-linker-combined'] = config.pop(f'{dataset}_data-ngame-category-linker')
            config[f'{dataset}_data-ngame-category-linker-combined']['path']['test']['data_info'] = save_file

            config_file = f'configs/beir/{dataset}_data-ngame-category-linker-combined.json'
            with open(config_file, 'w') as file:
                json.dump(config, file, indent=4)
        else:
            raise ValueError(f'Invalid type: {TYPE}')



