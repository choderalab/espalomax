from qcportal import FractalClient
import numpy as onp
from openff.toolkit.topology import Molecule
import h5py

def process_one_record(collection, record_name):
        record = collection.get_record(record_name, specification="default")
        entry = collection.get_entry(record_name)
        molecule = Molecule.from_qcschema(entry)
        smiles = molecule.to_smiles(mapped=True)
        trajectory = record.get_trajectory()
        u = onp.array([snapshot.properties.scf_total_energy for snapshot in trajectory])
        x = onp.stack([snapshot.get_molecule().geometry for snapshot in trajectory])
        f = onp.stack([snapshot.dict()["return_result"] for snapshot in trajectory])
        return smiles, x, u, f

def run(args):
    client = FractalClient()
    out = args.out
    if len(out) == 0:
        out = args.dataset.replace(" ", "") + ".hdf5"
    collection = client.get_collection("OptimizationDataset", args.dataset)
    record_names = list(collection.data.records)

    max_workers = args.max_workers
    if max_workers == 0:
        max_workers = None
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=max_workers)
    results = []
    for record_name in record_names:
        results.append(
            pool.submit(process_one_record, collection, record_name)
        )
    results = [result.result() for result in results]
    results = [result for result in results if result is not None]
    out = h5py.File(out, "w")

    for smiles, x, u, f in results:
        name = smiles.replace("/", "")
        if name not in out:
            group = out.create_group(name)
            group.create_dataset("smiles", data=[smiles])
            group.create_dataset("x", data=x)
            group.create_dataset("u", data=u)
            group.create_dataset("f", data=f)
        else:
            group = out[name]
            _x, _u, _f = group["x"], group["u"], group["f"]
            del group["x"], group["u"], group["f"]
            group.create_dataset("x", data=onp.concatenate([_x, x], 0))
            group.create_dataset("u", data=onp.concatenate([_u, u], 0))
            group.create_dataset("f", data=onp.concatenate([_f, f], 0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--dataset", type=str, default="OpenFF Gen 2 Opt Set 1 Roche")
    parser.add_argument("--max_workers", type=str, default=0)
    args = parser.parse_args()
    run(args)
