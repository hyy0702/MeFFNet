import os
import csv
import json
import random
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import math
import numpy as np
import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from learning_system.FFiNet_regressor import get_atom_features


# -------------------- utilities for conformer augmentation --------------------

def random_rotation_matrix():
    # uniform random rotation using quaternion method
    u1 = random.random()
    u2 = random.random()
    u3 = random.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    # convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*(q3*q3+q4*q4), 2*(q2*q3 - q1*q4), 2*(q2*q4 + q1*q3)],
        [2*(q2*q3 + q1*q4), 1 - 2*(q2*q2+q4*q4), 2*(q3*q4 - q1*q2)],
        [2*(q2*q4 - q1*q3), 2*(q3*q4 + q1*q2), 1 - 2*(q2*q2+q3*q3)]
    ], dtype=float)
    return R


# -------------------- CSV -> pseudo-assay reader --------------------
def read_tabular_assay(csv_path,
                       smiles_col_candidates=("smiles", "SMILES", "Smiles"),
                       target_col=None,
                       target_col_candidates=None,
                       domain="dataset",
                       min_group_size=8,
                       fallback_chunk=40,
                       std_filter=None,
                       seed=42,
                       verbose=True):
    if csv_path is None:
        raise ValueError("csv_path must be provided")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    # 读取 header
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []

    # detect smiles col
    smiles_col = None
    for c in smiles_col_candidates:
        if c in header:
            smiles_col = c
            break
    if smiles_col is None:
        for c in header:
            if 'smile' in c.lower():
                smiles_col = c
                break
    if smiles_col is None:
        raise ValueError(f"Cannot detect SMILES column in {csv_path}; header: {header}")

    # detect target col
    if target_col is None:
        if target_col_candidates is None:
            target_col_candidates = ["expt", "exp", "value",
                                     "measured log solubility", "measured log solubility in mols per litre",
                                     "logS", "logP", "pIC50", "class", "p_np"]
        found = None
        for c in header:
            lc = c.lower()
            for cand in target_col_candidates:
                if cand in lc:
                    found = c
                    break
            if found:
                break
        if found is None:
            raise ValueError("Cannot detect target column; pass target_col explicitly")
        target_col = found

    # parse rows
    records = []
    with open(csv_path, newline='') as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            smi = (row.get(smiles_col) or "").strip()
            raw = row.get(target_col, None)
            if not smi or raw is None or str(raw).strip() == "":
                continue
            s = str(raw).strip()
            # remove leading comparators
            if s[0] in ('<', '>', '='):
                s = s[1:].strip()

            try:
                val = float(s)
            except Exception:
                # 如果无法解析为数值，则跳过
                continue

            records.append({"smiles": smi, "target": val, "domain": domain, "affi_prefix": ""})

    if verbose:
        print(f"[read_tabular_assay] loaded {len(records)} molecules from {csv_path}")

    # group by Murcko scaffold
    scaffold_map = {}
    for rec in records:
        smi = rec["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            key = "NO_PARSER"
        else:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                key = Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold is not None else "NO_SCAFFOLD"
            except Exception:
                key = "NO_SCAFFOLD"
        scaffold_map.setdefault(key, []).append(rec)

    # keep large groups, pool small ones
    assay_dict = {}
    small_pool = []
    for k, v in scaffold_map.items():
        if len(v) >= min_group_size:
            aid = f"{k[:80]}/{len(v)}"
            assay_dict[aid] = v
        else:
            small_pool.extend(v)

    # chunk small_pool
    for i in range(0, len(small_pool), fallback_chunk):
        chunk = small_pool[i:i+fallback_chunk]
        aid = f"SMALLPOOL_{i//fallback_chunk}_{len(chunk)}"
        assay_dict[aid] = chunk

    # optional std filter and final pruning
    ligand_sets = {}
    for aid, ligs in assay_dict.items():
        vals = [x["target"] for x in ligs]
        if std_filter is not None and np.std(vals) <= std_filter:
            continue
        if len(ligs) < min_group_size:
            continue
        ligand_sets[aid] = ligs

    if verbose:
        print(f"[read_tabular_assay] kept {len(ligand_sets)} pseudo-assays (min_group_size={min_group_size}, fallback_chunk={fallback_chunk})")

    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}


def read_esol_assay(csv_path=None, **kwargs):
    if csv_path is None:
        csv_path = os.path.join("./datas/esol", "delaney.csv")
    return read_tabular_assay(csv_path, target_col=kwargs.pop("target_col", None), domain="esol", **kwargs)


def read_bace_assay(csv_path=None, **kwargs):
    if csv_path is None:
        csv_path = os.path.join("./datas/bace", "bace.csv")
    # 默认 BACE 在你的 CSV 中使用 'Class' 列
    return read_tabular_assay(csv_path, target_col=kwargs.pop("target_col", "Class"), domain="bace", **kwargs)


def read_bbbp_assay(csv_path=None, **kwargs):
    if csv_path is None:
        csv_path = os.path.join("./datas/bbbp", "bbbp.csv")
    # 默认 BBBP 在你的 CSV 中使用 'p_np' 列
    return read_tabular_assay(csv_path, target_col=kwargs.pop("target_col", "p_np"), domain="bbbp", **kwargs)


# -------------------- molecule -> Data object (with conformer augmentation) --------------------
def embed_conformers(mol_h, n_confs=5, use_uff=True, seed=None):
    # Ensure we have a fresh copy so we don't mix conformers across molecules accidentally
    m = Chem.Mol(mol_h)
    params = AllChem.ETKDG()
    if seed is not None:
        params.randomSeed = int(seed)
    try:
        ids = AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=params)
    except Exception:
        return None, []

    if use_uff:
        try:
            AllChem.UFFOptimizeMoleculeConfs(m)
        except Exception:
            pass

    return m, list(ids)


def process_assay(aid, entries, out_dir, n_confs=5, augment_num=1, jitter_translate=0.0, seed=42):
    processed = []
    rng = random.Random(seed)
    for entry in entries:
        smi = entry["smiles"]
        target_val = entry.get("target", None)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol_h = Chem.AddHs(mol)
        if mol_h.GetNumAtoms() == 0:
            continue
        if target_val is None:
            continue

        m_with_confs, conf_ids = embed_conformers(mol_h, n_confs=n_confs, use_uff=True, seed=rng.randint(0, 2**30))
        if m_with_confs is None or len(conf_ids) == 0:
            # try single conformer fallback
            try:
                m_temp = Chem.Mol(mol_h)
                st = AllChem.EmbedMolecule(m_temp, AllChem.ETKDG())
                if st != 0:
                    continue
                conf_ids = [m_temp.GetConformer().GetId()]
                m_with_confs = m_temp
            except Exception:
                continue

        # choose which conformers to emit (allow duplicates if augment_num > available confs)
        picks = [rng.choice(conf_ids) for _ in range(augment_num)] if augment_num > 1 else [rng.choice(conf_ids)]

        # common static molecule-level tensors
        feats = [get_atom_features(a) for a in m_with_confs.GetAtoms()]
        x = torch.tensor(feats, dtype=torch.float)
        edges = []
        for b in m_with_confs.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)

        triples = []
        for b in m_with_confs.GetBonds():
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            for nb in m_with_confs.GetAtomWithIdx(a1).GetNeighbors():
                c = nb.GetIdx()
                if c != a2:
                    triples.append([c, a1, a2])
            for nb in m_with_confs.GetAtomWithIdx(a2).GetNeighbors():
                c = nb.GetIdx()
                if c != a1:
                    triples.append([a1, a2, c])
        triple_index = torch.tensor(triples, dtype=torch.long).t().contiguous() if triples else torch.zeros((3,0), dtype=torch.long)

        quads = []
        for b in m_with_confs.GetBonds():
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            for na in m_with_confs.GetAtomWithIdx(a1).GetNeighbors():
                if na.GetIdx() == a2:
                    continue
                for nb in m_with_confs.GetAtomWithIdx(a2).GetNeighbors():
                    if nb.GetIdx() == a1:
                        continue
                    quads.append([na.GetIdx(), a1, a2, nb.GetIdx()])
        quadra_index = torch.tensor(quads, dtype=torch.long).t().contiguous() if quads else torch.zeros((4,0), dtype=torch.long)

        ne = edge_index.size(1)
        edge_attr = torch.ones(ne, dtype=torch.bool)
        y = torch.tensor([float(target_val)], dtype=torch.float32)
        batch = torch.zeros(x.size(0), dtype=torch.long)

        # create one Data per selected conformer, optionally rotate/translate as random augmentation
        for ci in picks:
            try:
                conf = m_with_confs.GetConformer(ci)
                coords = np.array(conf.GetPositions(), dtype=float)
            except Exception:
                continue

            # apply random rotation
            R = random_rotation_matrix()
            coords = coords.dot(R.T)
            # optional small random translation jitter
            if jitter_translate and jitter_translate > 0.0:
                t = rng.uniform(-jitter_translate, jitter_translate)
                coords = coords + t

            pos = torch.tensor(coords, dtype=torch.float)

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


# -------------------- split writer（含 molecule-level fallback） --------------------
def write_split(ligand_sets, out_dir, dataset_name="dataset", ratios=(0.8,0.1,0.1), seed=42):
    assay_ids = list(ligand_sets.keys())
    random.seed(seed)
    random.shuffle(assay_ids)
    n = len(assay_ids)

    # 如果 pseudo-assay 数太少，回退为 molecule-level 划分
    if n < 3:
        # flatten molecules
        all_items = []  # list of tuples (aid, idx)
        for aid, ligs in ligand_sets.items():
            for idx, rec in enumerate(ligs):
                all_items.append((aid, idx))
        random.shuffle(all_items)
        N = len(all_items)
        n_train = int(N * ratios[0])
        n_val = int(N * ratios[1])
        train_items = all_items[:n_train]
        val_items = all_items[n_train:n_train + n_val]
        test_items = all_items[n_train + n_val:]

        # create artificial assay lists that reference (aid, idx) pairs as JSON-serializable strings
        def items_to_ids(items):
            return [f"{a}||{i}" for (a, i) in items]

        split = {"train": items_to_ids(train_items), "valid": items_to_ids(val_items), "test": items_to_ids(test_items)}
        split_path = os.path.join(out_dir, f"{dataset_name}_split.json")
        with open(split_path, "w") as fo:
            json.dump(split, fo, indent=2)
        print(f"Saved molecule-level split json: {split_path} (train={len(train_items)} val={len(val_items)} test={len(test_items)})")
        return split_path

    # 否则按 pseudo-assay 划分（原有逻辑）
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = assay_ids[:n_train]
    valid = assay_ids[n_train:n_train + n_val]
    test = assay_ids[n_train + n_val:]
    split = {"train": train, "valid": valid, "test": test}
    split_path = os.path.join(out_dir, f"{dataset_name}_split.json")
    with open(split_path, "w") as fo:
        json.dump(split, fo, indent=2)
    print(f"Saved split json: {split_path} (train={len(train)} val={len(valid)} test={len(test)})")
    return split_path


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="esol", choices=["esol", "bace", "bbbp"], help="dataset to process")
    parser.add_argument("--csv", type=str, default=None, help="input csv file path (override default)")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    parser.add_argument("--min_group_size", type=int, default=12, help="minimum group size for scaffold grouping (esol default=12)")
    parser.add_argument("--fallback_chunk", type=int, default=40, help="chunk size for pooling small scaffolds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--n_confs", type=int, default=5, help="number of conformers to embed per molecule")
    parser.add_argument("--augment_num", type=int, default=1, help="how many augmented examples to emit per molecule (<= n_confs recommended)")
    parser.add_argument("--jitter_translate", type=float, default=0.05, help="max translation jitter applied after rotation (angstroms)")
    args = parser.parse_args()

    if args.dataset == 'esol':
        csv_path = args.csv or os.path.join('./datas/esol', 'delaney.csv')
        out_dir = args.out_dir or './esol_processed_aug'
        min_group_size = args.min_group_size or 12
        fallback_chunk = args.fallback_chunk or 40
        data = read_esol_assay(csv_path=csv_path, min_group_size=min_group_size, fallback_chunk=fallback_chunk)
    elif args.dataset == 'bace':
        csv_path = args.csv or os.path.join('./datas/bace', 'bace.csv')
        out_dir = args.out_dir or './bace_processed_aug'
        # 对 BACE/BBBP 推荐更小的分组/更细的 chunk
        min_group_size = args.min_group_size or 8
        fallback_chunk = args.fallback_chunk or 8
        data = read_bace_assay(csv_path=csv_path, min_group_size=min_group_size, fallback_chunk=fallback_chunk)
    elif args.dataset == 'bbbp':
        csv_path = args.csv or os.path.join('./datas/bbbp', 'bbbp.csv')
        out_dir = args.out_dir or './bbbp_processed_aug'
        min_group_size = args.min_group_size or 8
        fallback_chunk = args.fallback_chunk or 8
        data = read_bbbp_assay(csv_path=csv_path, min_group_size=min_group_size, fallback_chunk=fallback_chunk)
    else:
        raise ValueError(args.dataset)

    os.makedirs(out_dir, exist_ok=True)
    ligand_sets = data['ligand_sets']
    assays = list(ligand_sets.items())

    workers = args.num_workers if args.num_workers is not None else min(cpu_count(), 8)
    print(f"Found {len(assays)} assays, launching with {workers} workers (seed={args.seed})")

    # 使用多进程处理每个 pseudo-assay
    with Pool(processes=workers) as pool:
        fn = partial(process_assay, out_dir=out_dir, n_confs=args.n_confs, augment_num=args.augment_num, jitter_translate=args.jitter_translate, seed=args.seed)
        results = pool.starmap(fn, assays)

    print("======================================")
    for r in results:
        print("Processed", r)

    # 写 split json
    write_split(ligand_sets, out_dir, dataset_name=args.dataset, ratios=(0.8,0.1,0.1), seed=args.seed)
    print("All done! .pt files and split.json saved to:", out_dir)


if __name__ == '__main__':
    main()
















# import os
# import csv
# import json
# import random
# import argparse
# from functools import partial
# from multiprocessing import Pool, cpu_count
#
# import math
# import numpy as np
# import torch
# from torch_geometric.data import Data
#
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem.Scaffolds import MurckoScaffold
# from learning_system.FFiNet_regressor import get_atom_features
#
#
# # -------------------- CSV -> pseudo-assay reader --------------------
# def read_tabular_assay(csv_path,
#                        smiles_col_candidates=("smiles", "SMILES", "Smiles"),
#                        target_col=None,
#                        target_col_candidates=None,
#                        domain="dataset",
#                        min_group_size=8,
#                        fallback_chunk=40,
#                        std_filter=None,
#                        seed=42,
#                        verbose=True):
#     if csv_path is None:
#         raise ValueError("csv_path must be provided")
#     if not os.path.isfile(csv_path):
#         raise FileNotFoundError(csv_path)
#
#     # 读取 header
#     with open(csv_path, newline='') as fh:
#         reader = csv.DictReader(fh)
#         header = reader.fieldnames or []
#
#     # detect smiles col
#     smiles_col = None
#     for c in smiles_col_candidates:
#         if c in header:
#             smiles_col = c
#             break
#     if smiles_col is None:
#         for c in header:
#             if 'smile' in c.lower():
#                 smiles_col = c
#                 break
#     if smiles_col is None:
#         raise ValueError(f"Cannot detect SMILES column in {csv_path}; header: {header}")
#
#     # detect target col
#     if target_col is None:
#         if target_col_candidates is None:
#             target_col_candidates = ["expt", "exp", "value",
#                                      "measured log solubility", "measured log solubility in mols per litre",
#                                      "logS", "logP", "pIC50", "class", "p_np"]
#         found = None
#         for c in header:
#             lc = c.lower()
#             for cand in target_col_candidates:
#                 if cand in lc:
#                     found = c
#                     break
#             if found:
#                 break
#         if found is None:
#             raise ValueError("Cannot detect target column; pass target_col explicitly")
#         target_col = found
#
#     # parse rows
#     records = []
#     with open(csv_path, newline='') as fh:
#         rdr = csv.DictReader(fh)
#         for row in rdr:
#             smi = (row.get(smiles_col) or "").strip()
#             raw = row.get(target_col, None)
#             if not smi or raw is None or str(raw).strip() == "":
#                 continue
#             s = str(raw).strip()
#             # remove leading comparators
#             if s[0] in ('<', '>', '='):
#                 s = s[1:].strip()
#
#             try:
#                 val = float(s)
#             except Exception:
#                 # 如果无法解析为数值，则跳过
#                 continue
#
#             records.append({"smiles": smi, "target": val, "domain": domain, "affi_prefix": ""})
#
#     if verbose:
#         print(f"[read_tabular_assay] loaded {len(records)} molecules from {csv_path}")
#
#     # group by Murcko scaffold
#     scaffold_map = {}
#     for rec in records:
#         smi = rec["smiles"]
#         mol = Chem.MolFromSmiles(smi)
#         if mol is None:
#             key = "NO_PARSER"
#         else:
#             try:
#                 scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#                 key = Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold is not None else "NO_SCAFFOLD"
#             except Exception:
#                 key = "NO_SCAFFOLD"
#         scaffold_map.setdefault(key, []).append(rec)
#
#     # keep large groups, pool small ones
#     assay_dict = {}
#     small_pool = []
#     for k, v in scaffold_map.items():
#         if len(v) >= min_group_size:
#             aid = f"{k[:80]}/{len(v)}"
#             assay_dict[aid] = v
#         else:
#             small_pool.extend(v)
#
#     # chunk small_pool
#     for i in range(0, len(small_pool), fallback_chunk):
#         chunk = small_pool[i:i+fallback_chunk]
#         aid = f"SMALLPOOL_{i//fallback_chunk}_{len(chunk)}"
#         assay_dict[aid] = chunk
#
#     # optional std filter and final pruning
#     ligand_sets = {}
#     for aid, ligs in assay_dict.items():
#         vals = [x["target"] for x in ligs]
#         if std_filter is not None and np.std(vals) <= std_filter:
#             continue
#         if len(ligs) < min_group_size:
#             continue
#         ligand_sets[aid] = ligs
#
#     if verbose:
#         print(f"[read_tabular_assay] kept {len(ligand_sets)} pseudo-assays (min_group_size={min_group_size}, fallback_chunk={fallback_chunk})")
#
#     return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}
#
#
# def read_esol_assay(csv_path=None, **kwargs):
#     if csv_path is None:
#         csv_path = os.path.join("./datas/esol", "delaney.csv")
#     return read_tabular_assay(csv_path, target_col=kwargs.pop("target_col", None), domain="esol", **kwargs)
#
#
# def read_bace_assay(csv_path=None, **kwargs):
#     if csv_path is None:
#         csv_path = os.path.join("./datas/bace", "bace.csv")
#     # 默认 BACE 在你的 CSV 中使用 'Class' 列
#     return read_tabular_assay(csv_path, target_col=kwargs.pop("target_col", "Class"), domain="bace", **kwargs)
#
#
# def read_bbbp_assay(csv_path=None, **kwargs):
#     if csv_path is None:
#         csv_path = os.path.join("./datas/bbbp", "bbbp.csv")
#     # 默认 BBBP 在你的 CSV 中使用 'p_np' 列
#     return read_tabular_assay(csv_path, target_col=kwargs.pop("target_col", "p_np"), domain="bbbp", **kwargs)
#
#
# # -------------------- molecule -> Data object --------------------
# def process_assay(aid, entries, out_dir):
#     processed = []
#     for entry in entries:
#         smi = entry["smiles"]
#         target_val = entry.get("target", None)
#         if target_val is None:
#             continue
#
#         mol = Chem.MolFromSmiles(smi)
#         if mol is None:
#             continue
#
#         mol_h = Chem.AddHs(mol)
#         if mol_h.GetNumAtoms() == 0:
#             continue
#         try:
#             status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
#         except Exception:
#             # embed 失败跳过
#             continue
#         if status != 0:
#             continue
#
#         # 节点特征
#         feats = [get_atom_features(a) for a in mol_h.GetAtoms()]
#         x = torch.tensor(feats, dtype=torch.float)
#
#         # 边索引
#         edges = []
#         for b in mol_h.GetBonds():
#             i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
#             edges += [[i, j], [j, i]]
#         edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)
#
#         # triple_index
#         triples = []
#         for b in mol_h.GetBonds():
#             a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
#             for nb in mol_h.GetAtomWithIdx(a1).GetNeighbors():
#                 c = nb.GetIdx()
#                 if c != a2:
#                     triples.append([c, a1, a2])
#             for nb in mol_h.GetAtomWithIdx(a2).GetNeighbors():
#                 c = nb.GetIdx()
#                 if c != a1:
#                     triples.append([a1, a2, c])
#         triple_index = torch.tensor(triples, dtype=torch.long).t().contiguous() if triples else torch.zeros((3,0), dtype=torch.long)
#
#         # quadra_index
#         quads = []
#         for b in mol_h.GetBonds():
#             a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
#             for na in mol_h.GetAtomWithIdx(a1).GetNeighbors():
#                 if na.GetIdx() == a2:
#                     continue
#                 for nb in mol_h.GetAtomWithIdx(a2).GetNeighbors():
#                     if nb.GetIdx() == a1:
#                         continue
#                     quads.append([na.GetIdx(), a1, a2, nb.GetIdx()])
#         quadra_index = torch.tensor(quads, dtype=torch.long).t().contiguous() if quads else torch.zeros((4,0), dtype=torch.long)
#
#         # pos
#         try:
#             pos = torch.tensor(mol_h.GetConformer().GetPositions(), dtype=torch.float)
#         except Exception:
#             # 如果没有构型则跳过
#             continue
#
#         # edge_attr (保持布尔型占位)
#         ne = edge_index.size(1)
#         edge_attr = torch.ones(ne, dtype=torch.bool)
#
#         # y: 保存为 float（训练时再根据任务转换为 long/bce）
#         y = torch.tensor([float(target_val)], dtype=torch.float32)
#
#         # batch
#         batch = torch.zeros(x.size(0), dtype=torch.long)
#
#         data_obj = Data(
#             x=x,
#             edge_index=edge_index,
#             triple_index=triple_index,
#             quadra_index=quadra_index,
#             pos=pos,
#             edge_attr=edge_attr,
#             y=y,
#             batch=batch,
#             smiles=smi
#         )
#         processed.append(data_obj)
#
#     # 保存
#     fname = aid.replace('/', '_') + '.pt'
#     torch.save(processed, os.path.join(out_dir, fname))
#     return f"{aid}: {len(processed)} molecules"
#
#
# # -------------------- split writer（含 molecule-level fallback） --------------------
# def write_split(ligand_sets, out_dir, dataset_name="dataset", ratios=(0.8,0.1,0.1), seed=42):
#     assay_ids = list(ligand_sets.keys())
#     random.seed(seed)
#     random.shuffle(assay_ids)
#     n = len(assay_ids)
#
#     # 如果 pseudo-assay 数太少，回退为 molecule-level 划分
#     if n < 3:
#         # flatten molecules
#         all_items = []  # list of tuples (aid, idx)
#         for aid, ligs in ligand_sets.items():
#             for idx, rec in enumerate(ligs):
#                 all_items.append((aid, idx))
#         random.shuffle(all_items)
#         N = len(all_items)
#         n_train = int(N * ratios[0])
#         n_val = int(N * ratios[1])
#         train_items = all_items[:n_train]
#         val_items = all_items[n_train:n_train + n_val]
#         test_items = all_items[n_train + n_val:]
#
#         # create artificial assay lists that reference (aid, idx) pairs as JSON-serializable strings
#         def items_to_ids(items):
#             return [f"{a}||{i}" for (a, i) in items]
#
#         split = {"train": items_to_ids(train_items), "valid": items_to_ids(val_items), "test": items_to_ids(test_items)}
#         split_path = os.path.join(out_dir, f"{dataset_name}_split.json")
#         with open(split_path, "w") as fo:
#             json.dump(split, fo, indent=2)
#         print(f"Saved molecule-level split json: {split_path} (train={len(train_items)} val={len(val_items)} test={len(test_items)})")
#         return split_path
#
#     # 否则按 pseudo-assay 划分（原有逻辑）
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
#
#
#
# # -------------------- main --------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, default="esol", choices=["esol", "bace", "bbbp"], help="dataset to process")
#     parser.add_argument("--csv", type=str, default=None, help="input csv file path (override default)")
#     parser.add_argument("--out_dir", type=str, default=None, help="output directory")
#     parser.add_argument("--min_group_size", type=int, default=12, help="minimum group size for scaffold grouping (esol default=12)")
#     parser.add_argument("--fallback_chunk", type=int, default=40, help="chunk size for pooling small scaffolds")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--num_workers", type=int, default=None)
#     args = parser.parse_args()
#
#     if args.dataset == 'esol':
#         csv_path = args.csv or os.path.join('./datas/esol', 'delaney.csv')
#         out_dir = args.out_dir or './esol_processed'
#         min_group_size = args.min_group_size or 12
#         fallback_chunk = args.fallback_chunk or 40
#         data = read_esol_assay(csv_path=csv_path, min_group_size=min_group_size, fallback_chunk=fallback_chunk)
#     elif args.dataset == 'bace':
#         csv_path = args.csv or os.path.join('./datas/bace', 'bace.csv')
#         out_dir = args.out_dir or './bace_processed'
#         # 对 BACE/BBBP 推荐更小的分组/更细的 chunk
#         min_group_size = args.min_group_size or 8
#         fallback_chunk = args.fallback_chunk or 8
#         data = read_bace_assay(csv_path=csv_path, min_group_size=min_group_size, fallback_chunk=fallback_chunk)
#     elif args.dataset == 'bbbp':
#         csv_path = args.csv or os.path.join('./datas/bbbp', 'bbbp.csv')
#         out_dir = args.out_dir or './bbbp_processed'
#         min_group_size = args.min_group_size or 8
#         fallback_chunk = args.fallback_chunk or 8
#         data = read_bbbp_assay(csv_path=csv_path, min_group_size=min_group_size, fallback_chunk=fallback_chunk)
#     else:
#         raise ValueError(args.dataset)
#
#     os.makedirs(out_dir, exist_ok=True)
#     ligand_sets = data['ligand_sets']
#     assays = list(ligand_sets.items())
#
#     workers = args.num_workers if args.num_workers is not None else min(cpu_count(), 8)
#     print(f"Found {len(assays)} assays, launching with {workers} workers (seed={args.seed})")
#
#     # 使用多进程处理每个 pseudo-assay
#     with Pool(processes=workers) as pool:
#         fn = partial(process_assay, out_dir=out_dir)
#         results = pool.starmap(fn, assays)
#
#     print("======================================")
#     for r in results:
#         print("Processed", r)
#
#     # 写 split json
#     write_split(ligand_sets, out_dir, dataset_name=args.dataset, ratios=(0.8,0.1,0.1), seed=args.seed)
#     print("All done! .pt files and split.json saved to:", out_dir)
#
#
# if __name__ == '__main__':
#     main()
