import json
import os
import pandas as pd
import numpy as np

def _load_json(path:str):
    if os.path.exists(path):
        with open(path, "r") as f:
            data=json.load(f)
        return data
    else:
        return {}

def get_info(path:str):
    data = _load_json(path)

    df = pd.DataFrame(data['predictions'])

    df['final_pred'] = df["y_pred"].apply(lambda x: np.argmax(x))
    df['is_correct'] = df["y_true"] == df['final_pred']
    df['confidence'] = df["y_pred"].apply(lambda x: np.max(x))
    

    unique_labels = {p["y_true"] for p in data["predictions"]}
    num_unique = len(unique_labels)

    print(f"There are {num_unique} unique classes in the predictions.")

def _add_level(l:int = 1):
    level = ' ' * (4*(l-1)) +'-' * l
    return level
def get_struc(data:dict, l:int=1):
    if l == 1:
        print("JSON Structure ")
    #level = ' ' * (4*(l-1)) +'-' * l
    level = _add_level(l)
    for key, item in data.items():
        print(f'{level} {key}')
        if isinstance(item, dict):
            get_struc(item, l+1)
        else:
            item_level = _add_level(l+1)
            print(f'{item_level} {type(item)}')
        

if __name__ == "__main__":
    #data=_load_json("/home/stud/nemmler/retristyle/results/tta_inference/predictions/imagenet/test_r/densenet121_geometric_tpt_nviews64_seed71397589.json")
    data=_load_json('/home/stud/nemmler/retristyle/results/tta_inference/results/imagenet/test_r/densenet121_geometric_tpt_nviews8_seed71397589.json')
    get_struc(data)
