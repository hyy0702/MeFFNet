import os
import csv
import json
import random
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
import math
import pickle
import numpy as np
import signal
from contextlib import contextmanager

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    print("[Error] PyTorch or PyG not installed.")
    exit(1)

from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans

# 确保引用路径正确
from learning_system.FFiNet_regressor import get_atom_features
from dataset.load_dataset import pack_small_groups_by_similarity


# -------------------- Helper: Timeout Context --------------------
class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# -------------------- Helper: Fingerprints for K-Means --------------------
def get_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = []
    valid_indices = []
    for i, m in enumerate(mols):
        if m is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
                valid_indices.append(i)
            except:
                pass
    if len(fps) == 0:
        return np.array([]), []
    return np.array(fps), valid_indices


def split_large_group_by_clustering(records, group_name, max_size=50, seed=42):
    n_total = len(records)
    n_clusters = int(math.ceil(n_total / max_size))

    if n_clusters <= 1:
        return {group_name: records}

    smiles_list = [r['smiles'] for r in records]
    X, valid_indices = get_fingerprints(smiles_list)

    if len(X) < n_clusters:
        return {group_name: records}

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X)
    except Exception:
        return {group_name: records}

    new_groups = {}
    for local_idx, label in enumerate(labels):
        original_idx = valid_indices[local_idx]
        record = records[original_idx]
        sub_task_name = f"{group_name}_cluster_{label}"
        if sub_task_name not in new_groups:
            new_groups[sub_task_name] = []
        new_groups[sub_task_name].append(record)

    return new_groups


# -------------------- Label Parsing --------------------
def parse_label_classification(raw, minus_one_missing=False):
    if raw is None:
        return np.nan
    if isinstance(raw, (int, float)):
        if math.isnan(raw):
            return np.nan
        v = int(raw)
        if v == -1:
            return np.nan if minus_one_missing else 0.0
        if v in (0, 1):
            return float(v)
        return np.nan
    s = str(raw).strip()
    if s == "":
        return np.nan
    s_low = s.lower()
    if s_low in ("na", "nan", "none", "missing"):
        return np.nan
    if s_low in ("1", "true", "t", "positive", "pos", "active", "act", "+1"):
        return 1.0
    if s_low in ("0", "false", "f", "negative", "neg", "inactive", "inact", "-1", "-1.0"):
        if s_low == "-1":
            return np.nan if minus_one_missing else 0.0
        return 0.0
    try:
        val = float(s)
        if math.isnan(val):
            return np.nan
        ival = int(round(val))
        if ival in (0, 1):
            return float(ival)
        return np.nan
    except Exception:
        return np.nan


# -------------------- CSV Reader --------------------
def read_classification_csv_improved(csv_path,
                                     smiles_col_candidates=("smiles", "SMILES", "Smiles", "smi"),
                                     id_col_candidates=("mol_id", "id", "molecule_id", "name"),
                                     target_cols_override=None,
                                     minus_one_missing=False,
                                     domain="dataset",
                                     min_group_size=8,
                                     max_task_size=50,
                                     fallback_chunk=40,
                                     seed=42,
                                     verbose=True):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []

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
        raise ValueError(f"Cannot detect SMILES column in {csv_path}")

    id_col = None
    for c in id_col_candidates:
        if c in header:
            id_col = c
            break

    if target_cols_override:
        if isinstance(target_cols_override, str) and ',' in target_cols_override:
            target_cols = [t.strip() for t in target_cols_override.split(',') if t.strip()]
        else:
            if isinstance(target_cols_override, list):
                target_cols = target_cols_override
            else:
                target_cols = [target_cols_override]
    else:
        non_meta = [c for c in header if c not in (smiles_col, id_col)]
        # 简单清洗列名
        target_cols = [c.strip().replace(' ', '_').replace('/', '-') for c in non_meta]

    if verbose:
        print(f"[read] smiles={smiles_col}, targets={len(target_cols)} cols")

    records = []
    with open(csv_path, newline='') as fh:
        rdr = csv.DictReader(fh)
        raw_header = rdr.fieldnames
        clean_map = {c.strip().replace(' ', '_').replace('/', '-'): c for c in raw_header}

        for row in rdr:
            smi = (row.get(smiles_col) or "").strip()
            if not smi:
                continue
            mol_id = (row.get(id_col) or '').strip() if id_col else ''

            targets = []
            any_valid = False
            for tc in target_cols:
                raw_key = clean_map.get(tc, tc)
                raw_val = row.get(raw_key, None)
                val = parse_label_classification(raw_val, minus_one_missing=minus_one_missing)
                targets.append(val)
                if not (isinstance(val, float) and math.isnan(val)):
                    any_valid = True

            if not any_valid:
                continue

            rec = {'smiles': smi, 'targets': np.array(targets, dtype=float), 'mol_id': mol_id, 'domain': domain}
            records.append(rec)

    # Group by Scaffold
    scaffold_map = {}
    for rec in records:
        smi = rec['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            key = 'NO_PARSER'
        else:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                key = Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold is not None else 'NO_SCAFFOLD'
                if key == "": key = "GENERIC_ACYCLIC"
            except Exception:
                key = 'NO_SCAFFOLD'
        scaffold_map.setdefault(key, []).append(rec)

    # Splitting & Packing
    assay_dict = {}
    small_pool_groups = []
    sorted_scaffolds = sorted(scaffold_map.keys())

    for k in sorted_scaffolds:
        v = scaffold_map[k]
        if len(v) > max_task_size:
            sub_tasks = split_large_group_by_clustering(v, k, max_size=max_task_size, seed=seed)
            assay_dict.update(sub_tasks)
        elif len(v) >= min_group_size:
            aid = f"{k[:80]}/{len(v)}"
            assay_dict[aid] = v
        else:
            small_pool_groups.append(v)

    if len(small_pool_groups) > 0:
        if verbose:
            print(f"[read] Packing {len(small_pool_groups)} small groups by similarity...")
        packed_chunks = pack_small_groups_by_similarity(
            small_pool_groups,
            chunk_size=fallback_chunk,
            seed=seed
        )
        assay_dict.update(packed_chunks)

    final_ligand_sets = {}
    for aid, ligs in assay_dict.items():
        if len(ligs) < 4:
            continue
        final_ligand_sets[aid] = ligs

    return {
        'ligand_sets': final_ligand_sets,  # 聚类/打包后的数据 (对应 Clustered Split)
        'assays': list(final_ligand_sets.keys()),
        'target_names': target_cols,
        'raw_scaffolds': scaffold_map  # 原始骨架数据 (对应 Root Split)
    }


# -------------------- Convert to Single Task --------------------
def convert_to_single_task_format(ligand_sets, target_names, dataset_name, is_multitask=False, verbose=True):
    new_ligand_sets = {}
    expanded_map = {}
    dropped_counts = 0
    total_expanded = 0

    for aid, ligs in ligand_sets.items():
        if is_multitask:
            for t_idx, t_name in enumerate(target_names):
                new_entries = []
                for rec in ligs:
                    arr = rec.get('targets')
                    try:
                        v = float(arr[t_idx])
                    except:
                        continue
                    if math.isnan(v): continue

                    new_rec = dict(rec)
                    new_rec['target'] = v
                    new_rec['task_idx'] = t_idx
                    new_rec['task_name'] = t_name
                    new_entries.append(new_rec)

                if len(new_entries) < 4:
                    continue

                labels = [e['target'] for e in new_entries]
                n_pos = sum(l == 1.0 for l in labels)
                n_neg = sum(l == 0.0 for l in labels)

                if n_pos < 1 or n_neg < 1:
                    dropped_counts += 1
                    continue

                safe_t_name = t_name.replace('/', '-').replace(' ', '_')
                new_aid = f"{aid}__TASK{t_idx}__{safe_t_name}"
                new_ligand_sets[new_aid] = new_entries

                meta_key = f"{dataset_name}_{new_aid}"
                expanded_map[meta_key] = [new_aid, t_idx, t_name]
                total_expanded += 1

        else:
            t_idx = 0
            t_name = target_names[0] if len(target_names) > 0 else "default"
            new_entries = []
            for rec in ligs:
                arr = rec.get('targets')
                valid_val = None
                for i in range(len(arr)):
                    try:
                        v = float(arr[i])
                        if not math.isnan(v):
                            valid_val = v
                            break
                    except:
                        pass
                if valid_val is None: continue
                new_rec = dict(rec)
                new_rec['target'] = valid_val
                new_rec['task_idx'] = 0
                new_rec['task_name'] = t_name
                new_entries.append(new_rec)

            if len(new_entries) < 4: continue

            labels = [e['target'] for e in new_entries]
            n_pos = sum(l == 1.0 for l in labels)
            n_neg = sum(l == 0.0 for l in labels)
            if n_pos < 1 or n_neg < 1:
                dropped_counts += 1
                continue

            new_aid = aid
            new_ligand_sets[new_aid] = new_entries
            meta_key = f"{dataset_name}_{new_aid.replace('/', '_')}"
            expanded_map[meta_key] = [new_aid, 0, t_name]
            total_expanded += 1

    if verbose:
        mode = "Multitask-Expanded" if is_multitask else "Single-Task"
        print(f"[{mode}] Created {len(new_ligand_sets)} valid tasks.")
        print(f"  Dropped {dropped_counts} tasks due to pure-positive or pure-negative.")

    return new_ligand_sets, expanded_map


# -------------------- Process Classification (Final .pt) --------------------
def process_assay_classification(aid, entries, out_dir, dataset_name):
    processed = []
    safe_filename = aid.replace('/', '_')

    for entry in entries:
        smi = entry['smiles']

        # --- Label Handling ---
        y = None
        task_idx = entry.get('task_idx', 0)
        task_name = entry.get('task_name', "default")

        if 'target' in entry:  # Single task scalar
            y = torch.tensor([entry['target']], dtype=torch.float32)
        elif 'targets' in entry:  # Multi-task vector
            y = torch.tensor(entry['targets'], dtype=torch.float32)
        else:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        mol_h = Chem.AddHs(mol)
        if mol_h.GetNumAtoms() == 0: continue

        # --- 3D Conformer with Timeout ---
        try:
            with time_limit(15):
                params = AllChem.ETKDG()
                params.maxIterations = 800
                status = AllChem.EmbedMolecule(mol_h, params)

                if status != 0:
                    params.useRandomCoords = True
                    params.maxIterations = 1000
                    status = AllChem.EmbedMolecule(mol_h, params)

                if status != 0:
                    params.useRandomCoords = True
                    params.boxSizeMult = 2.0
                    status = AllChem.EmbedMolecule(mol_h, params)

            if status != 0: continue

        except TimeoutException:
            continue
        except Exception:
            continue

        feats = [get_atom_features(a) for a in mol_h.GetAtoms()]
        x = torch.tensor(feats, dtype=torch.float)
        edges = []
        for b in mol_h.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0),
                                                                                                      dtype=torch.long)

        triples = []
        for b in mol_h.GetBonds():
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            for nb in mol_h.GetAtomWithIdx(a1).GetNeighbors():
                c = nb.GetIdx()
                if c != a2: triples.append([c, a1, a2])
            for nb in mol_h.GetAtomWithIdx(a2).GetNeighbors():
                c = nb.GetIdx()
                if c != a1: triples.append([a1, a2, c])
        triple_index = torch.tensor(triples, dtype=torch.long).t().contiguous() if triples else torch.zeros((3, 0),
                                                                                                            dtype=torch.long)

        quads = []
        for b in mol_h.GetBonds():
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            for na in mol_h.GetAtomWithIdx(a1).GetNeighbors():
                if na.GetIdx() == a2: continue
                for nb in mol_h.GetAtomWithIdx(a2).GetNeighbors():
                    if nb.GetIdx() == a1: continue
                    quads.append([na.GetIdx(), a1, a2, nb.GetIdx()])
        quadra_index = torch.tensor(quads, dtype=torch.long).t().contiguous() if quads else torch.zeros((4, 0),
                                                                                                        dtype=torch.long)

        try:
            pos = torch.tensor(mol_h.GetConformer().GetPositions(), dtype=torch.float)
        except:
            continue

        batch = torch.zeros(x.size(0), dtype=torch.long)
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.bool)

        data_obj = Data(
            x=x,
            edge_index=edge_index,
            pos=pos,
            y=y,
            batch=batch,
            smiles=smi,
            triple_index=triple_index,
            quadra_index=quadra_index,
            edge_attr=edge_attr
        )

        data_obj.task_idx = torch.tensor([task_idx], dtype=torch.long)
        data_obj.task_name = task_name

        processed.append(data_obj)

    if len(processed) == 0:
        return f"{safe_filename}: Skipped (No valid 3D confs)"

    fname = safe_filename + '.pt'
    torch.save(processed, os.path.join(out_dir, fname))
    return f"{safe_filename}: {len(processed)} molecules"


# -------------------- Smart Split Logic --------------------
def write_split_classification_smart(ligand_sets, out_dir, dataset_name, is_multitask=False,
                                     ratios=(0.8, 0.1, 0.1), seed=42):
    family_map = {}
    total_mols_expanded = 0

    task_keys = list(ligand_sets.keys())

    for aid in task_keys:
        entries = ligand_sets[aid]
        n_mols = len(entries)

        # 1. Expanded ID (训练文件名)
        expanded_id = aid.replace('/', '_')

        # 2. Clustered ID (中间文件名)
        if "__TASK" in aid:
            clustered_name_raw = aid.split("__TASK")[0]
        else:
            clustered_name_raw = aid
        clustered_id = clustered_name_raw.replace('/', '_')

        # 3. Root ID (原始骨架名)
        if "_cluster_" in clustered_name_raw:
            root_name_raw = clustered_name_raw.split("_cluster_")[0]
        elif clustered_name_raw.startswith("SMALLPOOL"):
            root_name_raw = clustered_name_raw
        elif "/" in clustered_name_raw:
            root_name_raw = clustered_name_raw.split("/")[0]
        else:
            root_name_raw = clustered_name_raw
        root_id = root_name_raw.replace('/', '_')

        if root_id not in family_map:
            family_map[root_id] = {'expanded': [], 'clustered': set()}

        family_map[root_id]['expanded'].append(expanded_id)
        family_map[root_id]['clustered'].add(clustered_id)
        total_mols_expanded += n_mols

    family_stats = []
    for root, data in family_map.items():
        total_m = sum(len(ligand_sets[t_raw]) for t_raw in task_keys if t_raw.replace('/', '_') in data['expanded'])
        family_stats.append({"root": root, "data": data, "total_mols": total_m})

    splits = {
        'root': {'train': [], 'valid': [], 'test': []},
        'clustered': {'train': [], 'valid': [], 'test': []},
        'expanded': {'train': [], 'valid': [], 'test': []}
    }

    curr_counts = {'train': 0, 'valid': 0, 'test': 0}
    target_test = total_mols_expanded * ratios[2]
    target_valid = total_mols_expanded * ratios[1]

    if is_multitask:
        if "toxcast" in dataset_name.lower():
            HUGE_THRESHOLD = 80000  # 针对 ToxCast 的特供阈值
        else:
            HUGE_THRESHOLD = 3000
    else:
        HUGE_THRESHOLD = 150

    random.seed(seed)
    random.shuffle(family_stats)

    rest_families = []

    def assign_to_split(split_name, item):
        splits['root'][split_name].append(item['root'])
        splits['clustered'][split_name].extend(list(item['data']['clustered']))
        splits['expanded'][split_name].extend(item['data']['expanded'])
        curr_counts[split_name] += item['total_mols']

    for item in family_stats:
        if item["total_mols"] > HUGE_THRESHOLD:
            assign_to_split('train', item)
        else:
            rest_families.append(item)

    for item in rest_families:
        if curr_counts['test'] < target_test:
            assign_to_split('test', item)
        elif curr_counts['valid'] < target_valid:
            assign_to_split('valid', item)
        else:
            assign_to_split('train', item)

    print(f"[{dataset_name}] Smart Split Stats (Seed={seed}):")
    print(f"  Total Mols (Expanded): {total_mols_expanded}")
    for k in ['train', 'valid', 'test']:
        n_root = len(splits['root'][k])
        n_clus = len(splits['clustered'][k])
        n_exp = len(splits['expanded'][k])
        print(f"  {k.capitalize():<5}: {n_root:<4} roots | {n_clus:<4} clusters | {n_exp:<5} tasks")

    # Save 1: Expanded
    path_exp = os.path.join(out_dir, f"{dataset_name}_split.json")
    with open(path_exp, "w") as f:
        json.dump(splits['expanded'], f, indent=2)
    print(f"  Saved Training Split   -> {path_exp}")

    # Save 2: Clustered
    path_clus = os.path.join(out_dir, f"{dataset_name}_clustered_split.json")
    with open(path_clus, "w") as f:
        json.dump(splits['clustered'], f, indent=2)
    print(f"  Saved Clustered Split  -> {path_clus}")

    # Save 3: Root (always save)
    path_root = os.path.join(out_dir, f"{dataset_name}_root_scaffold_split.json")
    with open(path_root, "w") as f:
        json.dump(splits['root'], f, indent=2)
    print(f"  Saved Root Split       -> {path_root}")

    return path_exp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--is_multitask", action='store_true', help="If true, expand tasks like Tox21")
    parser.add_argument("--min_group_size", type=int, default=6)
    parser.add_argument("--max_task_size", type=int, default=50)
    parser.add_argument("--fallback_chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_cols", type=str, default=None)
    parser.add_argument("--minus_one_missing", action='store_true')
    args = parser.parse_args()

    out_dir = args.out_dir or f"./{args.dataset}_processed"
    os.makedirs(out_dir, exist_ok=True)
    workers = args.num_workers if args.num_workers else min(cpu_count(), 8)

    print(f"--- Processing {args.dataset} ---")

    # 1. Read & Group
    data = read_classification_csv_improved(
        args.csv,
        target_cols_override=args.target_cols,
        minus_one_missing=args.minus_one_missing,
        domain=args.dataset,
        min_group_size=args.min_group_size,
        max_task_size=args.max_task_size,
        fallback_chunk=args.fallback_chunk,
        seed=args.seed
    )
    ligand_sets = data['ligand_sets']  # (Clustered)
    raw_scaffold_map = data.get('raw_scaffolds', {})  # (Root)
    target_names = data.get('target_names', [])

    # 2. Expand Tasks
    final_ligand_sets, expanded_map = convert_to_single_task_format(
        ligand_sets,
        target_names,
        dataset_name=args.dataset,
        is_multitask=args.is_multitask
    )

    # -------------------------------------------------------------
    # 3. Generate .pt files (Phase 1: Expanded -> 用于 FFiNet 训练)
    # 对应 _split.json
    # -------------------------------------------------------------
    print(f"\n[Phase 1] Generating Expanded .pt files for training ({len(final_ligand_sets)} tasks)...")
    final_assays_list = list(final_ligand_sets.items())

    with Pool(processes=workers) as pool:
        fn = partial(process_assay_classification, out_dir=out_dir, dataset_name=args.dataset)
        results = pool.starmap(fn, final_assays_list)

    # Filter failed tasks
    valid_expanded_sets = {}
    n_skipped = 0
    for i, res_str in enumerate(results):
        task_key = final_assays_list[i][0]
        if "Skipped" in res_str:
            n_skipped += 1
        else:
            valid_expanded_sets[task_key] = final_ligand_sets[task_key]

    print(f"Phase 1 Done. {n_skipped} skipped. Valid expanded tasks: {len(valid_expanded_sets)}")

    # -------------------------------------------------------------
    # 4. Generate .pt files (Phase 2: Clustered -> 对应 _clustered_split.json)
    # -------------------------------------------------------------
    print(f"\n[Phase 2] Generating Clustered .pt files...")
    clustered_tasks_list = []
    # 遍历 ligand_sets，注意要清洗 key 里的 '/'
    for k, v in ligand_sets.items():
        clean_key = k.replace('/', '_')
        clustered_tasks_list.append((clean_key, v))

    with Pool(processes=workers) as pool:
        fn = partial(process_assay_classification, out_dir=out_dir, dataset_name=args.dataset)
        results_clus = pool.starmap(fn, clustered_tasks_list)
    print(f"Phase 2 Done.")

    # -------------------------------------------------------------
    # 5. Generate .pt files (Phase 3: Root -> 对应 _root_scaffold_split.json)
    # -------------------------------------------------------------
    print(f"\n[Phase 3] Generating Root Scaffold .pt files...")
    root_tasks_list = []

    # (A) 大骨架
    for scaffold_smiles, entries in raw_scaffold_map.items():
        clean_key = scaffold_smiles.replace('/', '_')
        root_tasks_list.append((clean_key, entries))

    # (B) SmallPool (视为 Root)
    for k, v in ligand_sets.items():
        if k.startswith("SMALLPOOL"):
            clean_key = k.replace('/', '_')
            root_tasks_list.append((clean_key, v))

    with Pool(processes=workers) as pool:
        fn = partial(process_assay_classification, out_dir=out_dir, dataset_name=args.dataset)
        results_root = pool.starmap(fn, root_tasks_list)
    print(f"Phase 3 Done.")

    # -------------------------------------------------------------
    # 6. Generate Splits & Metadata
    # -------------------------------------------------------------
    # 使用 valid_expanded_sets 进行划分，逻辑会自动推导父级关系
    write_split_classification_smart(
        valid_expanded_sets,
        out_dir,
        dataset_name=args.dataset,
        is_multitask=args.is_multitask,
        seed=args.seed
    )

    meta = {
        'target_names': target_names,
        'expanded_map': expanded_map,
        'mode': "multitask" if args.is_multitask else "single_task"
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as fo:
        json.dump(meta, fo, indent=2)

    # 7. Save ligand_sets.pkl
    # 合并所有层级数据，方便 write_splits_csv.py 查找
    save_sets = valid_expanded_sets.copy()

    # Merge Clustered
    for k, v in ligand_sets.items():
        save_sets[k.replace('/', '_')] = v

    # Merge Root
    for aid, entries in root_tasks_list:
        save_sets[aid] = entries

    import pickle
    with open(os.path.join(out_dir, 'ligand_sets.pkl'), 'wb') as f:
        pickle.dump(save_sets, f)

    print(f"\nAll done! Output in: {out_dir}")


if __name__ == '__main__':
    main()