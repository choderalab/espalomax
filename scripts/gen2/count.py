import numpy as np
import pickle

def run():
    import os
    import pickle
    base_path = "../data/qca_optimization/data/"
    paths = os.listdir(base_path)
    paths = [base_path + path for path in paths]
    data = []
    for path in paths:
        _data = pickle.load(open(path, "rb"))
        data.append(_data)
    cache = []

    for g, _, __ in data:
        cache.append((
            int(g.homograph.n_node),
            len(g.heterograph["bond"]["idxs"]),
            len(g.heterograph["angle"]["idxs"]),
            len(g.heterograph["proper"]["idxs"]) + len(g.heterograph["improper"]["idxs"]),
        ))

    pickle.dump(cache, open("count.pkl", "wb"))


if __name__ == "__main__":
    run()
