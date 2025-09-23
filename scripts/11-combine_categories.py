import json, pandas as pd, re, ast
from tqdm.auto import tqdm

def get_dictionary(text:str):
    m = re.findall(r'\{[^{}]*\}', text)
    if len(m): return m[-1]
    else: return '{}'

def get_pairs(text:str):
    pattern = r'"(.*?)"\s*:\s*(\d+)'
    pairs = re.findall(pattern, text)
    return [(k, int(v)) for k,v in pairs]

def fix_unicode(text:str):
    text = re.sub(r'\\u\{([0-9a-fA-F]{1,4})\}', '', text)
    text = re.sub(r'\\u([0-9a-fA-F]{0,4})', '', text)
    return text

def replace_quotes(text:str):
    return re.sub(r'\"([a-zA-Z]+)\"[^:]', r"'\1'", text)

if __name__ == "__main__":
    d = "outputs/51_finetune-llama-for-category-generation/"

    all_categories = dict()
    for i in tqdm(range(6)):
        fname = f"{d}/category-00{i}.csv"
        df = pd.read_csv(fname)

        for k,v in zip(df['identifier'], df['text']):
            try:
                text = get_dictionary(v)
                pairs = get_pairs(text)
                if len(pairs) == 0: continue
            except Exception as e:
                raise ValueError(f"Invalid dictionary: {v}, with error {e}")

            all_categories[k] = {p:q for p,q in pairs}

    with open(f"{d}/all_categories.json", "w") as file:
        json.dump(all_categories, file, indent=4)

