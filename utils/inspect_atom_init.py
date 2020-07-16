import json

with open("../gen_data/atom_init.json") as f:
    data = json.load(f)
    for key, val in data.items():
        print(key, len(val))
