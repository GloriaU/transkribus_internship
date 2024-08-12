import os
import jsonlines
import numpy as np
import json

splits = [0.8, 0.1, 0.1]
folders = ['train', 'test', 'validation']

files = [[], [], []]

## get file data
with jsonlines.open(f"out/metadata.jsonl") as reader:
    for d in reader:
        s = np.random.choice(3, p=splits)
        files[s].append(d)
        
        file = d["file_name"]

        try:
            os.rename(f"out/{file}", f"genara_data/{folders[s]}/{file}")
        except:
            print(file)

with open("genara_data/train/metadata.jsonl", 'w') as f:
    for d in files[0]:
        f.write(json.dumps(d))
        f.write('\n')
        
with open("genara_data/test/metadata.jsonl", 'w') as f:
    for d in files[1]:
        f.write(json.dumps(d))
        f.write('\n')
        
with open("genara_data/validation/metadata.jsonl", 'w') as f:
    for d in files[2]:
        f.write(json.dumps(d))
        f.write('\n')
