import os
import torch
import random
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from multiprocessing import Pool, cpu_count
from functools import partial

from dataset.load_dataset import read_lipo_assay
from learning_system.FFiNet_regressor import get_atom_features


class Args:
    def __init__(self):
        self.no_fep_lig = False
        self.seed = 42
        self.num_workers = None

def process_assay(aid, entries, out_dir):
    processed = []
    for entry in entries:
        smi = entry["smiles"]
        pic50 = entry["pic50_exp"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mol_h = Chem.AddHs(mol)
        if mol_h.GetNumAtoms() == 0:
            continue
        try:
            status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        except Exception as e:
            # 任何 Embed 错误都跳过
            continue
        if status != 0:
            continue

        try:
            AllChem.ComputeGasteigerCharges(mol_h)
        except Exception:
            # 极少数分子可能计算失败，跳过或忽略
            continue

        # 节点特征
        feats = [get_atom_features(a) for a in mol_h.GetAtoms()]
        x = torch.tensor(feats, dtype=torch.float)

        # 边索引
        edges = []
        for b in mol_h.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)

        # triple_index
        triples = []
        for b in mol_h.GetBonds():
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            for nb in mol_h.GetAtomWithIdx(a1).GetNeighbors():
                c = nb.GetIdx()
                if c != a2: triples.append([c, a1, a2])
            for nb in mol_h.GetAtomWithIdx(a2).GetNeighbors():
                c = nb.GetIdx()
                if c != a1: triples.append([a1, a2, c])
        triple_index = torch.tensor(triples, dtype=torch.long).t().contiguous() if triples else torch.zeros((3,0), dtype=torch.long)

        # quadra_index
        quads = []
        for b in mol_h.GetBonds():
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            for na in mol_h.GetAtomWithIdx(a1).GetNeighbors():
                if na.GetIdx()==a2: continue
                for nb in mol_h.GetAtomWithIdx(a2).GetNeighbors():
                    if nb.GetIdx()==a1: continue
                    quads.append([na.GetIdx(), a1, a2, nb.GetIdx()])
        quadra_index = torch.tensor(quads, dtype=torch.long).t().contiguous() if quads else torch.zeros((4,0), dtype=torch.long)

        # pos
        pos = torch.tensor(mol_h.GetConformer().GetPositions(), dtype=torch.float)

        # edge_attr
        ne = edge_index.size(1)
        edge_attr = torch.ones(ne, dtype=torch.bool)

        # y
        y = torch.tensor([float(pic50)], dtype=torch.float32)

        # batch
        batch = torch.zeros(x.size(0), dtype=torch.long)

        data_obj = Data(
            x=x,
            edge_index=edge_index,
            triple_index=triple_index,
            quadra_index=quadra_index,
            pos=pos,
            edge_attr=edge_attr,
            y=y,
            batch=batch,
            smiles=smi
        )
        processed.append(data_obj)

    # 保存
    fname = aid.replace('/', '_') + '.pt'
    torch.save(processed, os.path.join(out_dir, fname))
    return f"{aid}: {len(processed)} molecules"

# def write_split(ligand_sets, out_dir, dataset_name="lipo", ratios=(0.8,0.1,0.1), seed=42):
#     assay_ids = list(ligand_sets.keys())
#     random.seed(seed)
#     random.shuffle(assay_ids)
#     n = len(assay_ids)
#     n_train = int(n * ratios[0])
#     n_val = int(n * ratios[1])
#     train = assay_ids[:n_train]
#     valid = assay_ids[n_train:n_train + n_val]
#     test = assay_ids[n_train + n_val:]
#     split = {"train": train, "valid": valid, "test": test}
#     split_path = os.path.join(out_dir, f"{dataset_name}_split.json")
#     with open(split_path, "w") as fo:
#         json.dump(split, fo, indent=2)
#     print(f"Saved split json: {split_path} (train={len(train)} val={len(valid)} test={len(test)})")
#     return split_path

from esol_processed import write_split, write_metadata_from_ligand_sets


def main():
    args = Args()
    data = read_lipo_assay(csv_path=None, min_group_size=40, fallback_chunk=60, max_task_size=80)
    ligand_sets = data["ligand_sets"]
    out_dir = "./lipo_processed"
    os.makedirs(out_dir, exist_ok=True)

    assays = list(ligand_sets.items())
    workers = args.num_workers if getattr(args, "num_workers", None) else min(cpu_count(), 8)
    print(f"Found {len(assays)} assays, launching with {workers} workers (seed={args.seed})")

    with Pool(processes=cpu_count()) as pool:
        fn = partial(process_assay, out_dir=out_dir)
        # map each (aid, entries) pair
        results = pool.starmap(fn, assays)

    # 打印每个 assay 处理结果
    print("======================================")
    for r in results:
        print("Processed", r)

    # write split json to out_dir root
    write_split(ligand_sets, out_dir, dataset_name="lipo", seed=args.seed)
    write_metadata_from_ligand_sets(ligand_sets, out_dir, dataset_name="lipo")
    print("All done! .pt files and split.json saved to:", out_dir)


if __name__ == "__main__":
    main()
