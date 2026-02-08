import os
import torch
import random
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from multiprocessing import Pool, cpu_count
from functools import partial

from dataset.load_dataset import read_esol_assay
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
        except Exception:
            # 任何 Embed 错误都跳过
            continue
        if status != 0:
            continue

        # try:
        #     AllChem.ComputeGasteigerCharges(mol_h)
        # except Exception:
        #     # 极少数分子可能计算失败，跳过或忽略
        #     continue

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

# def write_split(ligand_sets, out_dir, dataset_name="esol", ratios=(0.8,0.1,0.1), seed=42):
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

def write_split(ligand_sets, out_dir, dataset_name, ratios=(0.8, 0.1, 0.1), seed=42):
    # 1. 整理骨架
    scaffold_map = {}
    for raw_aid in ligand_sets.keys():
        clean_aid = raw_aid.replace('/', '_')

        # 【关键修复】
        if "_cluster_" in raw_aid:
            # 聚类后的子任务：归属于同一个根骨架
            root = raw_aid.split("_cluster_")[0]
        elif raw_aid.startswith("SMALLPOOL"):
            # 杂类池：每一个 Chunk 视为一个独立的骨架单位！
            # 不要把所有 SMALLPOOL 归为一类
            root = raw_aid
        elif "/" in raw_aid:
            # 普通骨架分组
            root = raw_aid.split("/")[0]
        else:
            root = raw_aid

        if root not in scaffold_map:
            scaffold_map[root] = []
        scaffold_map[root].append( (raw_aid, clean_aid) )

    # 2. 统计
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

    # 3. 初始化分配容器
    train_tasks = []
    valid_tasks = []
    test_tasks = []

    curr_train_mols = 0
    curr_valid_mols = 0
    curr_test_mols = 0

    target_test = total_mols_all * ratios[2]
    target_valid = total_mols_all * ratios[1]

    # 巨型骨架阈值
    HUGE_THRESHOLD = 150

    # 4. 策略分配
    random.seed(seed)
    random.shuffle(scaffold_stats)

    rest_scaffolds = []

    # 第一轮：强制分配巨型骨架到 Train (仅针对真正的单一骨架)
    for item in scaffold_stats:
        clean_keys = [t[1] for t in item["tasks"]]
        mols = item["n_mols"]

        if mols > HUGE_THRESHOLD:
            train_tasks.extend(clean_keys)
            curr_train_mols += mols
        else:
            rest_scaffolds.append(item)

    # 第二轮：用剩下的中小骨架(含SMALLPOOL)，优先填满 Test 和 Valid
    for item in rest_scaffolds:
        clean_keys = [t[1] for t in item["tasks"]]
        mols = item["n_mols"]

        # 优先填 Test
        if curr_test_mols < target_test:
            test_tasks.extend(clean_keys)
            curr_test_mols += mols
        # 然后填 Valid
        elif curr_valid_mols < target_valid:
            valid_tasks.extend(clean_keys)
            curr_valid_mols += mols
        # 剩下的给 Train
        else:
            train_tasks.extend(clean_keys)
            curr_train_mols += mols

    print(f"[{dataset_name}] Smart Strict Scaffold Split Stats (Seed={seed}):")
    print(f"  Total Mols: {total_mols_all}")
    print(f"  Train: {len(train_tasks)} tasks (~{curr_train_mols} mols, {curr_train_mols/total_mols_all:.1%})")
    print(f"  Valid: {len(valid_tasks)} tasks (~{curr_valid_mols} mols, {curr_valid_mols/total_mols_all:.1%})")
    print(f"  Test : {len(test_tasks)} tasks (~{curr_test_mols} mols, {curr_test_mols/total_mols_all:.1%})")

    if curr_valid_mols < 50 or curr_test_mols < 50:
         print("  [WARNING] Validation/Test set is too small. Try a different seed!")

    split = {"train": train_tasks, "valid": valid_tasks, "test": test_tasks}
    split_path = os.path.join(out_dir, f"{dataset_name}_split.json")
    with open(split_path, "w") as fo:
        json.dump(split, fo, indent=2)
    return split_path

def write_metadata_from_ligand_sets(ligand_sets, out_dir, dataset_name):
    expanded_map = {}
    target_names = []
    for aid in ligand_sets.keys():
        task_name = f"{dataset_name}_{aid.replace('/', '_')}"
        target_names.append(task_name)
        expanded_map[task_name] = [aid, 0, task_name]

    meta = {"expanded_map": expanded_map, "target_names": target_names}
    meta_path = os.path.join(out_dir, "metadata.json")
    try:
        with open(meta_path, "w") as fo:
            json.dump(meta, fo, indent=2)
        print(f"Saved metadata.json: {meta_path} (targets={len(target_names)})")
    except Exception as e:
        print(f"Failed to write metadata.json to {meta_path}: {e}")
    return meta_path

def main():
    args = Args()
    data = read_esol_assay(csv_path=None, min_group_size=16, fallback_chunk=40, max_task_size=50)
    ligand_sets = data["ligand_sets"]
    out_dir = "./esol_processed"
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
    write_split(ligand_sets, out_dir, dataset_name="esol", seed=args.seed)

    # 写 metadata.json（供 main_reg.py 使用）
    write_metadata_from_ligand_sets(ligand_sets, out_dir, dataset_name="esol")

    print("All done! .pt files, split.json and metadata.json saved to:", out_dir)


if __name__ == "__main__":
    main()
