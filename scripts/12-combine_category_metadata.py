import json, os

def load_category(fname):
    with open(fname) as file:
        o = json.load(file)
    return o


if __name__ == '__main__':
    files = ['/data/outputs/mogicX/51_finetune-llama-for-category-generation/outputs/all_categories.json', 
            '/data/share/from_deepak/gpt_generations/MSMARCOGenerations5/categories.json', 
            '/data/share/from_deepak/msmarco/category_gpt_oracle/all_categories.json']

    categories = dict()
    for file in files:
        o = load_category(file)
        print(f'Number of categories: {len(o)}')
        categories.update(o)

    save_file = '/data/share/from_deepak/msmarco/category_gpt_labels/final_category.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    with open(save_file, 'w') as file:
        json.dump(categories, file, indent=4)

