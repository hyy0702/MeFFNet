import os
import torch
import random
import json
import math
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ==========================================
# 1. 配置参数
# ==========================================
class Args:
    def __init__(self):
        self.seed = 42
        self.num_workers = 8  # 根据服务器CPU核数调整


# QM9 的 12 个目标属性
TARGET_COLS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']


# ==========================================
# 2. 特征提取 (无 Gasteiger 电荷版)
# ==========================================
def get_atom_features(atom):
    features = []
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Other']
    features += [1 if atom.GetSymbol() == t else 0 for t in atom_types]

    hybridization = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    features += [int(atom.GetHybridization() == h) for h in hybridization]

    features += [
        atom.GetIsAromatic(), atom.IsInRing(), atom.GetFormalCharge(),
        atom.GetTotalNumHs(), atom.GetDegree()
    ]

    # 直接使用形式电荷
    features.append(float(atom.GetFormalCharge()))

    # Crippen LogP
    contrib = 0.0
    try:
        val = Crippen.ContribTable[Crippen.GetAtomCorr(atom)]
        if not math.isnan(val) and not math.isinf(val):
            contrib = float(val)
    except:
        contrib = 0.0
    features.append(contrib)

    while len(features) < 66:
        features.append(0)
    return features[:66]


# ==========================================
# 3. 辅助函数：聚类与相似度合并
# ==========================================
def get_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = []
    valid_indices = []
    for i, m in enumerate(mols):
        if m is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_indices.append(i)
    return np.array(fps), valid_indices


def split_large_group_by_clustering(records, group_name, max_size=50, seed=42):
    n_total = len(records)
    n_clusters = int(math.ceil(n_total / max_size))
    if n_clusters <= 1:
        return {group_name: records}

    smiles_list = [r["smiles"] for r in records]
    X, valid_indices = get_fingerprints(smiles_list)

    if len(X) < n_clusters:
        random.seed(seed)
        random.shuffle(records)
        chunks = [records[i::n_clusters] for i in range(n_clusters)]
        return {f"{group_name}_part{i}": chunk for i, chunk in enumerate(chunks) if chunk}

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)

    new_groups = {}
    for local_idx, label in enumerate(labels):
        original_idx = valid_indices[local_idx]
        record = records[original_idx]
        sub_task_name = f"{group_name}_cluster_{label}"
        if sub_task_name not in new_groups:
            new_groups[sub_task_name] = []
        new_groups[sub_task_name].append(record)
    return new_groups


def get_group_fingerprint_vector(group_records):
    valid_fps = []
    for r in group_records:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            valid_fps.append(arr)
    if not valid_fps:
        return np.zeros(1024)
    return np.mean(np.stack(valid_fps), axis=0)


def pack_small_groups_by_similarity(small_pool_groups, chunk_size, seed=42):
    if not small_pool_groups:
        return {}

    print(f"  [Packing] Intelligent merging of {len(small_pool_groups)} small groups by similarity...")
    group_vectors = []
    for grp in small_pool_groups:
        vec = get_group_fingerprint_vector(grp)
        group_vectors.append(vec)
    group_vectors = np.stack(group_vectors)

    sim_matrix = cosine_similarity(group_vectors)
    rng = np.random.RandomState(seed)
    assigned_mask = np.zeros(len(small_pool_groups), dtype=bool)

    final_chunks = {}
    chunk_idx = 0

    while not np.all(assigned_mask):
        remaining_indices = np.where(~assigned_mask)[0]
        if len(remaining_indices) == 0: break

        seed_idx = rng.choice(remaining_indices)
        current_indices = [seed_idx]
        current_mol_count = len(small_pool_groups[seed_idx])
        assigned_mask[seed_idx] = True

        while current_mol_count < chunk_size:
            candidates = np.where(~assigned_mask)[0]
            if len(candidates) == 0: break

            sims = sim_matrix[seed_idx, candidates]
            best_local_idx = np.argmax(sims)
            best_global_idx = candidates[best_local_idx]

            current_indices.append(best_global_idx)
            current_mol_count += len(small_pool_groups[best_global_idx])
            assigned_mask[best_global_idx] = True

        merged_data = []
        for g_idx in current_indices:
            merged_data.extend(small_pool_groups[g_idx])

        task_name = f"SMALLPOOL_SIM_{chunk_idx}_{len(merged_data)}"
        final_chunks[task_name] = merged_data
        chunk_idx += 1

    return final_chunks


# ==========================================
# 4. 数据读取 (生成两套任务集)
# ==========================================
def read_qm9_assay_combined(csv_path, min_group_size=20, fallback_chunk=50, max_task_size=100, seed=42):
    print(f"[QM9] Loading from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}, {}, {}

    # 1. Global Normalization
    print("[QM9] Performing Global Normalization...")
    stats = {}
    for col in TARGET_COLS:
        if col not in df.columns: continue
        vals = df[col].values.astype(float)
        mean = np.mean(vals)
        std = np.std(vals)
        stats[col] = (mean, std)
        df[col] = (vals - mean) / (std + 1e-8)

    # 2. Group by Scaffold
    scaffold_map = {}
    smi_col = 'smiles' if 'smiles' in df.columns else 'SMILES1'

    print("[QM9] Generating scaffolds...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smi = row.get(smi_col, "")
        if not isinstance(smi, str) or not smi: continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            key = Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold else "NO_SCAFFOLD"
            if key == "": key = "GENERIC_ACYCLIC"
        except:
            key = "NO_SCAFFOLD"

        record = {
            "smiles": smi,
            "domain": "qm9",
            "affi_prefix": ""
        }
        for col in TARGET_COLS:
            if col in row:
                record[col] = float(row[col])

        if key not in scaffold_map:
            scaffold_map[key] = []
        scaffold_map[key].append(record)

    # 3. Create Task Sets
    expanded_sets = {}  # For Meta-Learning (12 tasks per scaffold)
    original_sets = {}  # For Baseline (1 task per scaffold)

    # Helpers for processing groups
    small_pool_groups = []

    sorted_scaffolds = sorted(scaffold_map.keys())

    for k in sorted_scaffolds:
        v = scaffold_map[k]

        # Split huge clusters
        current_groups = []
        if len(v) > max_task_size:
            sub_dict = split_large_group_by_clustering(v, k, max_size=max_task_size, seed=seed)
            for sub_name, sub_data in sub_dict.items():
                current_groups.append((sub_name, sub_data))
        else:
            current_groups.append((k, v))

        for g_name, g_data in current_groups:
            if len(g_data) >= min_group_size:
                # --- A. Create Expanded Tasks ---
                for col in TARGET_COLS:
                    task_data = []
                    for r in g_data:
                        new_r = r.copy()
                        new_r['pic50_exp'] = r.get(col, 0.0)
                        new_r['task_type'] = col
                        task_data.append(new_r)

                    safe_name = g_name[:50] if len(g_name) > 50 else g_name
                    aid = f"{safe_name}_{col}"
                    expanded_sets[aid] = task_data

                # --- B. Create Original Task ---
                task_data_orig = []
                for r in g_data:
                    new_r = r.copy()
                    new_r['pic50_exp'] = r.get('u0', 0.0)  # Dummy target
                    task_data_orig.append(new_r)

                safe_name = g_name[:50] if len(g_name) > 50 else g_name
                aid_orig = f"{safe_name}_original"
                original_sets[aid_orig] = task_data_orig
            else:
                small_pool_groups.append(g_data)

    # 4. Handle Small Pool
    if len(small_pool_groups) > 0:
        sim_chunks = pack_small_groups_by_similarity(small_pool_groups, chunk_size=fallback_chunk, seed=seed)

        for chunk_name, chunk_data in sim_chunks.items():
            # --- A. Expanded ---
            for col in TARGET_COLS:
                task_data = []
                for r in chunk_data:
                    new_r = r.copy()
                    new_r['pic50_exp'] = r.get(col, 0.0)
                    new_r['task_type'] = col
                    task_data.append(new_r)
                aid = f"{chunk_name}_{col}"
                expanded_sets[aid] = task_data

            # --- B. Original ---
            task_data_orig = []
            for r in chunk_data:
                new_r = r.copy()
                new_r['pic50_exp'] = r.get('u0', 0.0)
                task_data_orig.append(new_r)
            aid_orig = f"{chunk_name}_original"
            original_sets[aid_orig] = task_data_orig

    print(f"[QM9] Generated {len(expanded_sets)} Expanded Tasks and {len(original_sets)} Original Tasks.")
    return expanded_sets, original_sets, stats


# ==========================================
# 5. 图处理 (通用)
# ==========================================
def process_assay(data_tuple, out_dir):
    aid, entries = data_tuple

    processed = []
    for entry in entries:
        smi = entry["smiles"]
        # 对于展开任务，这里是归一化后的单属性值
        # 对于原始骨架任务，这里是一个 dummy 值 (u0)
        pic50 = entry["pic50_exp"]

        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        mol_h = Chem.AddHs(mol)
        if mol_h.GetNumAtoms() == 0: continue

        try:
            status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        except:
            continue
        if status != 0: continue

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

        pos = torch.tensor(mol_h.GetConformer().GetPositions(), dtype=torch.float)
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.bool)

        y = torch.tensor([float(pic50)], dtype=torch.float32)
        batch = torch.zeros(x.size(0), dtype=torch.long)

        # 保存所有属性
        extra_props = {col: torch.tensor([entry.get(col, 0.0)], dtype=torch.float32) for col in TARGET_COLS}

        data_obj = Data(
            x=x, edge_index=edge_index, triple_index=triple_index, quadra_index=quadra_index,
            pos=pos, edge_attr=edge_attr, y=y, batch=batch, smiles=smi,
            **extra_props
        )
        processed.append(data_obj)

    fname = aid.replace('/', '_') + '.pt'
    torch.save(processed, os.path.join(out_dir, fname))
    return f"Processed {aid}"


# ==========================================
# 6. 划分函数
# ==========================================
def write_split(ligand_sets, out_dir, filename_prefix, ratios=(0.8, 0.1, 0.1), seed=42):
    scaffold_map = {}

    # 如果是 Expanded (有多属性)，需要还原骨架名进行分组
    is_expanded = (filename_prefix == "qm9")

    for aid in ligand_sets.keys():
        root = aid
        if is_expanded:
            for p in TARGET_COLS:
                suffix = f"_{p}"
                if aid.endswith(suffix):
                    root = aid[:-len(suffix)]
                    break

        if root not in scaffold_map:
            scaffold_map[root] = []
        clean_aid = aid.replace('/', '_')
        scaffold_map[root].append((aid, clean_aid))

    scaffold_stats = []
    total_mols_all = 0
    for root, task_pairs in scaffold_map.items():
        n_mols = sum([len(ligand_sets[t[0]]) for t in task_pairs])
        scaffold_stats.append({
            "root": root,
            "n_mols": n_mols,
            "tasks": task_pairs
        })
        total_mols_all += n_mols

    train_tasks, valid_tasks, test_tasks = [], [], []
    curr_train, curr_valid, curr_test = 0, 0, 0

    target_test = total_mols_all * ratios[2]
    target_valid = total_mols_all * ratios[1]

    random.seed(seed)
    random.shuffle(scaffold_stats)

    for item in scaffold_stats:
        task_keys = [t[1] for t in item["tasks"]]
        mols = item["n_mols"]

        if curr_test < target_test:
            test_tasks.extend(task_keys)
            curr_test += mols
        elif curr_valid < target_valid:
            valid_tasks.extend(task_keys)
            curr_valid += mols
        else:
            train_tasks.extend(task_keys)
            curr_train += mols

    split_name = f"{filename_prefix}_split.json"
    print(
        f"[{filename_prefix}] Split generated: Train={len(train_tasks)} Valid={len(valid_tasks)} Test={len(test_tasks)} tasks")

    split = {"train": train_tasks, "valid": valid_tasks, "test": test_tasks}
    split_path = os.path.join(out_dir, split_name)
    with open(split_path, "w") as fo:
        json.dump(split, fo, indent=2)


def write_metadata(all_keys, out_dir):
    expanded_map = {}
    target_names = []
    for aid in all_keys:
        # 兼容性前缀
        prefix = "qm9" if not aid.endswith("_original") else "qm9_original"
        task_name = f"{prefix}_{aid.replace('/', '_')}"
        target_names.append(task_name)
        expanded_map[task_name] = [aid, 0, task_name]

    meta = {"expanded_map": expanded_map, "target_names": target_names}
    with open(os.path.join(out_dir, "metadata.json"), "w") as fo:
        json.dump(meta, fo, indent=2)


# ==========================================
# 7. 主程序
# ==========================================
def main():
    args = Args()
    csv_file = "./datas/qm9/qm9.csv"
    out_dir = "./qm9_processed_v2"

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return

    # 1. 读取并生成两套任务集合
    expanded_sets, original_sets, stats = read_qm9_assay_combined(
        csv_path=csv_file, min_group_size=20, fallback_chunk=50, max_task_size=100, seed=args.seed
    )

    # 2. 保存全局统计量
    global_stats_path = os.path.join(os.path.dirname(csv_file), "qm9_global_stats.json")
    with open(global_stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Global stats saved.")

    # 3. 合并所有任务进行图处理
    os.makedirs(out_dir, exist_ok=True)
    all_assays = list(expanded_sets.items()) + list(original_sets.items())
    print(f"Starting multiprocessing for {len(all_assays)} total files with {args.num_workers} workers...")

    with Pool(processes=args.num_workers) as pool:
        fn = partial(process_assay, out_dir=out_dir)
        # 使用 tqdm 显示进度
        list(tqdm(pool.imap(fn, all_assays), total=len(all_assays)))

    # 4. 分别生成两个 Split 文件
    print("\nGenerating splits...")
    # FFiNet 用的 split (qm9_split.json)
    write_split(expanded_sets, out_dir, "qm9", seed=args.seed)

    # 对比用的 split (qm9_original_split.json)
    write_split(original_sets, out_dir, "qm9_original", seed=args.seed)

    # 5. 生成统一的 metadata
    # 把两个 sets 的 key 合并写进 metadata，确保无论用哪个 split 都能查到
    all_keys = list(expanded_sets.keys()) + list(original_sets.keys())
    write_metadata(all_keys, out_dir)

    print(f"\nAll done! Data saved to {out_dir}")
    print(f"- Expanded Tasks: {len(expanded_sets)}")
    print(f"- Original Tasks: {len(original_sets)}")


if __name__ == "__main__":
    main()