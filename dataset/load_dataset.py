import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import math, os
from tqdm import tqdm
import random
import csv
from collections import OrderedDict
import json, pickle
from sklearn.cluster import KMeans

absolute_path = os.path.abspath(__file__)
DATA_PATH = "/".join(absolute_path.split("/")[:-2] + ["datas"])


def read_BDB_per_assay(args):
    data_dir = f"{DATA_PATH}/BDB/polymer"
    assays = []
    ligand_sets = {}
    split_cnt = 0
    means = []

    if args.no_fep_lig:
        fep_datas, _ = read_FEP_SET()
        fep_lig_set = set()
        for assay in fep_datas["ligand_sets"].values():
            for lig_info in assay:
                fep_lig_set.add(lig_info["smiles"])

    for target_name in tqdm(list(os.listdir(data_dir))):
        for assay_file in os.listdir(os.path.join(data_dir, target_name)):
            assay_name = target_name + "/" + assay_file
            entry_assay = "_".join(assay_file.split("_")[:2])
            affi_idx = int(assay_file[-5])
            ligands = []
            affis = []
            file_lines = list(open(os.path.join(data_dir, target_name, assay_file), "r").readlines())
            for i, line in enumerate(file_lines):
                line = line.strip().split("\t")
                affi_prefix = ""
                pic50_exp = line[8 + affi_idx].strip()
                if pic50_exp.startswith(">") or pic50_exp.startswith("<"):
                    continue
                    # affi_prefix = pic50_exp[0]
                    # pic50_exp = pic50_exp[1:]
                try:
                    pic50_exp = -math.log10(float(pic50_exp))
                except:
                    print("error ic50s:", pic50_exp)
                    continue
                smiles = line[1]
                if args.no_fep_lig:
                    if smiles in fep_lig_set:
                        print(smiles, "in fep")
                        continue
                affis.append(pic50_exp)
                ligand_info = {
                    "affi_idx": affi_idx,
                    "affi_prefix": affi_prefix,
                    "smiles": smiles,
                    "pic50_exp": pic50_exp,
                    "domain": "bdb"
                }
                ligands.append(ligand_info)
            pic50_exp_list = [x["pic50_exp"] for x in ligands]
            pic50_std = np.std(pic50_exp_list)
            if pic50_std <= 0.2:
                continue
            if len(ligands) < 20:
                continue
            means.append(np.mean([x["pic50_exp"] for x in ligands]))
            ligand_sets[assay_name] = ligands

    print(np.mean(means))
    print("split_cnt:", split_cnt)
    return {"ligand_sets": ligand_sets,
            "assays": list(ligand_sets.keys())}


def read_BDB_IC50():
    data_dir = f"{DATA_PATH}/BDB_baseline"
    ligand_sets = {}
    means = []

    for file_name in tqdm(list(os.listdir(data_dir))):
        assay_name = file_name
        affi_idx = 1
        ligands = []
        affis = []
        file_lines = list(open(os.path.join(data_dir, file_name), "r").readlines())
        for i, line in enumerate(file_lines):
            line = line.strip().split("\t")
            affi_prefix = ""
            pic50_exp = line[8 + affi_idx].strip()
            pic50_exp = float(pic50_exp) - 9

            smiles = line[1]
            affis.append(pic50_exp)
            ligand_info = {
                "affi_idx": affi_idx,
                "affi_prefix": affi_prefix,
                "smiles": smiles,
                "pic50_exp": pic50_exp,
                "domain": "bdb"
            }
            ligands.append(ligand_info)
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std == 0.0:
            continue
        if len(ligands) < 20:
            continue
        means.append(np.mean([x["pic50_exp"] for x in ligands]))
        ligand_sets[assay_name] = ligands

    return {"ligand_sets": ligand_sets,
            "assays": list(ligand_sets.keys())}


def read_gdsc():
    data_file = f"{DATA_PATH}/gdsc/data_dict.pkl"
    ligand_sets = pickle.load(open(data_file, "rb"))
    ligand_sets_new = {}
    for k, v in ligand_sets.items():
        ligand_sets_new[k] = v
        for ligand_info in v:
            ligand_info['pic50_exp'] = -math.log10(math.exp(ligand_info['pic50_exp']))
    print(np.mean([len(x) for x in ligand_sets_new.values()]))
    return {"ligand_sets": ligand_sets_new,
            "assays": list(ligand_sets_new.keys())}


def read_FEP_SET():
    datas = json.load(open(f"{DATA_PATH}/FEP/fep_data_final_norepeat_nocharge.json", "r"))
    ligand_sets = {}
    task2opls4 = {}
    pic50s_all = []
    rmse_all = []
    for k, v in datas.items():
        ligands = []
        opls4_res = []
        errors = []
        for ligand_info in v:
            pic50_exp = -float(ligand_info["exp_dg"]) / 1.379 - 9
            opls4 = -float(ligand_info["pred_dg"]) / 1.379 - 9
            errors.append(pic50_exp - opls4)
            opls4_res.append(opls4)
            smiles = ligand_info["smiles"]
            ligands.append({
                "affi_prefix": "",
                "smiles": smiles,
                "pic50_exp": pic50_exp,
                "domain": "fep"
            })
            pic50s_all.append(pic50_exp)
        ligand_sets[k] = ligands
        task2opls4[k] = np.array(opls4_res)
        rmse = np.sqrt(np.mean(np.square(errors)))
        r2 = np.corrcoef(opls4_res, [x["pic50_exp"] for x in ligands])[0, 1]
        rmse_all.append(r2)
    print("rmse_FEP+(OPLS4)", np.mean(rmse_all))
    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}, task2opls4


def read_kiba():
    ligand_list = []
    ligands_dict = json.load(open(f"{DATA_PATH}/DeepDTA/data/kiba/ligands_can.txt"), object_pairs_hook=OrderedDict)
    for ligand_id, smiles in ligands_dict.items():
        ligand_list.append((ligand_id, smiles))
    Y = pickle.load(open(f"{DATA_PATH}/DeepDTA/data/kiba/Y", "rb"), encoding='bytes').transpose()
    ligand_sets = {}
    stds = []
    for assay_idx in range(Y.shape[0]):
        affis = Y[assay_idx]
        ligands = []
        pic50s = []
        for i, affi in enumerate(affis):
            if not np.isnan(affi) and affi < 10000:
                ligand_id, smiles = ligand_list[i]
                pic50s.append((affi - 11.72) + -2.24)
                ligand_info = {
                    "affi_prefix": "",
                    "smiles": smiles,
                    "ligand_id": ligand_id,
                    "pic50_exp": (affi - 11.72) + -2.24
                }
                ligands.append(ligand_info)
        if len(ligands) < 20:
            continue
        ligand_sets[f"kiba_{assay_idx}"] = ligands

    assay_id_dicts_new = {}
    for assay_id, ligands in ligand_sets.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.5:
            continue
        if len(ligands) < 50:
            continue
        assay_id_dicts_new[assay_id] = ligands
        stds.append(pic50_std)
    print("stds", np.mean(stds), len(stds))
    return {"ligand_sets": assay_id_dicts_new, "assays": list(assay_id_dicts_new.keys())}


def read_davis():
    ligand_list = []
    protein_list = []
    ligands_dict = json.load(open(f"{DATA_PATH}/DeepDTA/data/davis/ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins_dict = json.load(open(f"{DATA_PATH}/DeepDTA/data/davis/proteins.txt"), object_pairs_hook=OrderedDict)
    for ligand_id, smiles in ligands_dict.items():
        ligand_list.append((ligand_id, smiles))
    for protein_id, seq in proteins_dict.items():
        protein_list.append((protein_id, seq))
    Y = pickle.load(open(f"{DATA_PATH}/DeepDTA/data/davis/Y", "rb"), encoding='bytes').transpose()
    ligand_sets = {}

    stds = []
    for assay_idx in range(Y.shape[0]):
        affis = Y[assay_idx]
        assay_name, seq = protein_list[assay_idx]
        ligands = []
        pic50s = []
        for i, affi in enumerate(affis):
            if not np.isnan(affi) and affi < 10000:
                ligand_id, smiles = ligand_list[i]
                pic50s.append(-math.log10(affi))
                ligand_info = {
                    "affi_prefix": "",
                    "smiles": smiles,
                    "ligand_id": ligand_id,
                    "pic50_exp": -math.log10(affi)
                }
                ligands.append(ligand_info)
        if len(ligands) < 20:
            continue
        stds.append(len(ligands))
        ligand_sets[f"davis_{assay_idx}"] = ligands

    print("stds", np.mean(stds), len(stds))
    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}


def read_fsmol_assay(split="train", train_phase=1):
    cache_file = f"{DATA_PATH}/fsmol/{split}_cache.pkl"
    if os.path.exists(cache_file):
        datas = pickle.load(open(cache_file, "rb"))
        for k, v in datas["ligand_sets"].items():
            ligands_new = []
            for item in v:
                try:
                    ligands_new.append({
                        "smiles": item["SMILES"],
                        "pic50_exp": eval(item["LogRegressionProperty"]),
                        "domain": "fsmol"
                    })
                except:
                    pass
            datas["ligand_sets"][k] = ligands_new
        return datas


def read_chembl_assay(args):
    datas = csv.reader(open(f"{DATA_PATH}/chembl/chembl_processed_chembl32.csv", "r"),
                       delimiter=',')
    assay_id_dicts = {}

    if args.no_fep_lig:
        fep_datas, _ = read_FEP_SET()
        fep_lig_set = set()
        for assay in fep_datas["ligand_sets"].values():
            for lig_info in assay:
                fep_lig_set.add(lig_info["smiles"])

    # kd_assay_set = set()
    for line in datas:
        unit = line[7]
        if unit == "%":
            continue
        assay_id = "{}_{}_{}".format(line[11], line[7], line[8]).replace("/", "_")
        if assay_id not in assay_id_dicts:
            assay_id_dicts[assay_id] = []
        smiles = line[13]
        assay_type = line[9]
        bao_endpoint = line[4]
        bao_format = line[10]
        std_type = line[8]
        # if std_type.lower() != "kd":
        #     continue
        unit = line[7]
        std_rel = line[5]
        if args.no_fep_lig:
            if smiles in fep_lig_set:
                continue

        if std_rel != "=":
            continue
        is_does = unit in ['ug.mL-1', 'ug ml-1', 'mg.kg-1', 'mg kg-1',
                           'mg/L', 'ng/ml', 'mg/ml', 'ug kg-1', 'mg/kg/day', 'mg kg-1 day-1',
                           "10'-4 ug/ml", 'M kg-1', "10'-6 ug/ml", 'ng/L', 'pmg kg-1', "10'-8mg/ml",
                           'ng ml-1', "10'-3 ug/ml", "10'-1 ug/ml", ]
        pic50_exp = -math.log10(float(line[6]))
        affi_prefix = line[5]
        ligand_info = {
            "assay_type": std_type,
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": affi_prefix,
            "is_does": is_does,
            "chembl_assay_type": assay_type,
            "bao_endpoint": bao_endpoint,
            "bao_format": bao_format,
            "unit": unit,
            "domain": "chembl"
        }
        assay_id_dicts[assay_id].append(ligand_info)

    # print(list(kd_assay_set))
    # exit()
    assay_id_dicts_new = {}
    for assay_id, ligands in assay_id_dicts.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.2:
            continue
        if len(ligands) < 20:
            continue
        assay_id_dicts_new[assay_id] = ligands

    return {"ligand_sets": assay_id_dicts_new, "assays": list(assay_id_dicts_new.keys())}


def get_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = []
    valid_indices = []
    for i, m in enumerate(mols):
        if m is not None:
            # 使用 Morgan 指纹 (ECFP4), 半径2, 1024位
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            fps.append(np.array(fp))
            valid_indices.append(i)
    return np.array(fps), valid_indices


def split_large_group_by_clustering(records, group_name, max_size=50, seed=42):
    """
    如果一个 Scaffold 组太大，使用 K-Means 将其拆分为多个子任务。
    这样做既增加了任务数量，又保证了子任务内部的化学相似性。
    """
    n_total = len(records)
    # 计算需要拆分成几份
    n_clusters = int(math.ceil(n_total / max_size))

    # 如果这组数据太少或计算出的簇为1，直接返回原样
    if n_clusters <= 1:
        return {group_name: records}

    print(f"  [Splitting] Group '{group_name}' has {n_total} mols. Splitting into {n_clusters} clusters...")

    smiles_list = [r["smiles"] for r in records]
    X, valid_indices = get_fingerprints(smiles_list)

    # 如果有效的指纹太少，无法聚类，随机拆分
    if len(X) < n_clusters:
        random.shuffle(records)
        chunks = [records[i::n_clusters] for i in range(n_clusters)]
        return {f"{group_name}_part{i}": chunk for i, chunk in enumerate(chunks) if chunk}

    # 执行 K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)

    new_groups = {}

    # 根据聚类标签重新分组
    # 注意：X 只对应 valid_indices，需要映射回原始 records
    # 为简单起见，这里忽略 RDKit 解析失败的极少数分子（它们不会在 X 中）

    for local_idx, label in enumerate(labels):
        original_idx = valid_indices[local_idx]
        record = records[original_idx]

        sub_task_name = f"{group_name}_cluster_{label}"
        if sub_task_name not in new_groups:
            new_groups[sub_task_name] = []
        new_groups[sub_task_name].append(record)

    return new_groups


from rdkit import DataStructs
from sklearn.metrics.pairwise import cosine_similarity

def get_group_fingerprint_vector(group_records):
    """
    计算一组分子的平均指纹向量。
    """
    valid_fps = []
    for r in group_records:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol:
            # 使用 Morgan 指纹, 半径2, 1024位 (与聚类保持一致)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            # 转为 numpy array
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            valid_fps.append(arr)

    if not valid_fps:
        return np.zeros(1024)

    # 计算均值向量
    return np.mean(np.stack(valid_fps), axis=0)

def pack_small_groups_by_similarity(small_pool_groups, chunk_size, seed=42):
    """
    【高级策略】基于化学相似度，贪心地将小骨架组合并成大任务。
    """
    if not small_pool_groups:
        return {}

    print(f"  [Packing] Intelligent merging of {len(small_pool_groups)} small groups by similarity...")

    # 1. 计算每个组的特征向量
    group_vectors = []
    for idx, grp in enumerate(small_pool_groups):
        vec = get_group_fingerprint_vector(grp)
        group_vectors.append(vec)

    group_vectors = np.stack(group_vectors)  # shape: [N_groups, 1024]

    # 2. 计算所有组之间的相似度矩阵 (Cosine Similarity)
    # 值越大越相似
    sim_matrix = cosine_similarity(group_vectors)

    # 3. 贪心装箱
    rng = np.random.RandomState(seed)

    # 标记哪些组已经被分配了
    assigned_mask = np.zeros(len(small_pool_groups), dtype=bool)
    n_groups = len(small_pool_groups)

    final_chunks = {}
    chunk_idx = 0

    while not np.all(assigned_mask):
        # A. 挑选种子：从未分配的组里随机选一个作为当前箱子的“核心”
        remaining_indices = np.where(~assigned_mask)[0]
        if len(remaining_indices) == 0:
            break

        seed_idx = rng.choice(remaining_indices)

        current_indices = [seed_idx]
        current_mol_count = len(small_pool_groups[seed_idx])
        assigned_mask[seed_idx] = True

        # B. 填箱子：寻找与当前箱子核心最相似的邻居
        while current_mol_count < chunk_size:
            # 拿到剩余未分配的组的索引
            candidates = np.where(~assigned_mask)[0]
            if len(candidates) == 0:
                break

            # 计算 candidates 与当前种子(seed_idx)的相似度
            sims = sim_matrix[seed_idx, candidates]

            # 找到最相似的那个
            best_local_idx = np.argmax(sims)
            best_global_idx = candidates[best_local_idx]

            # 加入箱子
            current_indices.append(best_global_idx)
            current_mol_count += len(small_pool_groups[best_global_idx])
            assigned_mask[best_global_idx] = True

        # C. 组装数据
        merged_data = []
        for g_idx in current_indices:
            merged_data.extend(small_pool_groups[g_idx])

        # D. 保存任务
        task_name = f"SMALLPOOL_SIM_{chunk_idx}_{len(merged_data)}"
        final_chunks[task_name] = merged_data
        chunk_idx += 1

    return final_chunks


def read_tabular_assay(csv_path,
                       smiles_col_candidates=("smiles", "SMILES", "Smiles"),
                       target_col=None,
                       target_col_candidates=None,
                       domain="dataset",
                       min_group_size=8,
                       fallback_chunk=40,
                       max_task_size=50,  # 最大任务容量，超过则拆分
                       std_filter=None,
                       seed=42,
                       verbose=True):
    """
    通用 CSV -> pseudo-assay reader.
    增加了大任务自动拆分逻辑 (K-Means)。
    """
    if csv_path is None:
        raise ValueError("csv_path must be provided")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    # 1. 读取 Header
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
        raise ValueError(f"Cannot detect SMILES column in {csv_path}; header: {header}")

    if target_col is None:
        if target_col_candidates is None:
            target_col_candidates = ["expt", "exp", "value", "measured log solubility", "logSolubility", "logS", "logP",
                                     "pIC50", "gap"]
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

    # 2. 读取数据
    records = []
    with open(csv_path, newline='') as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            smi = (row.get(smiles_col) or "").strip()
            raw = row.get(target_col, None)
            if not smi or raw is None or str(raw).strip() == "":
                continue
            s = str(raw).strip()
            if s[0] in ('<', '>', '='):
                s = s[1:].strip()
            try:
                val = float(s)
            except Exception:
                continue
            records.append({"smiles": smi, "pic50_exp": float(val), "domain": domain, "affi_prefix": ""})

    if verbose:
        print(f"[read_tabular_assay] loaded {len(records)} molecules from {csv_path}")

    # 3. 按 Murcko Scaffold 初步分组
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
                if key == "":
                    key = "GENERIC_ACYCLIC"
            except Exception:
                key = "NO_SCAFFOLD"
        scaffold_map.setdefault(key, []).append(rec)

    # 4. 构建任务 (修复泄露的核心部分)
    assay_dict = {}

    # 暂存小骨架组，而不是打散的分子
    small_pool_groups = []

    sorted_scaffolds = sorted(scaffold_map.keys())

    for k in sorted_scaffolds:
        v = scaffold_map[k]

        if len(v) > max_task_size:
            # 大任务：聚类拆分
            sub_tasks = split_large_group_by_clustering(v, k, max_size=max_task_size, seed=seed)
            assay_dict.update(sub_tasks)

        elif len(v) >= min_group_size:
            # 中等任务：直接保留
            aid = f"{k[:80]}/{len(v)}"
            assay_dict[aid] = v

        else:
            # 小任务：整组存入，不打散
            small_pool_groups.append(v)

    # 5. 基于相似度智能合并
    if len(small_pool_groups) > 0:
        sim_chunks = pack_small_groups_by_similarity(
            small_pool_groups,
            chunk_size=fallback_chunk,
            seed=seed
        )
        assay_dict.update(sim_chunks)

    # 6. 最终清理
    ligand_sets = {}
    means = []
    for aid, ligs in assay_dict.items():
        vals = [x["pic50_exp"] for x in ligs]
        if std_filter is not None and np.std(vals) <= std_filter:
            continue
        ligand_sets[aid] = ligs
        means.append(np.mean(vals))

    if verbose:
        print(f"[read_tabular_assay] Final processed: {len(ligand_sets)} tasks created.")
        print(f"  (Criteria: min_group={min_group_size}, max_task={max_task_size}, chunk={fallback_chunk})")

    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}



def read_esol_assay(csv_path=None, **kwargs):
    if csv_path is None:
        csv_path = os.path.join(DATA_PATH, "ESOL", "delaney.csv")
    return read_tabular_assay(csv_path, target_col="logSolubility", domain="esol", **kwargs)


def read_freesolv_assay(csv_path=None, **kwargs):
    if csv_path is None:
        csv_path = os.path.join(DATA_PATH, "freesolv", "freesolv.csv")
    return read_tabular_assay(csv_path,
                              target_col='freesolv',
                              domain="freesolv",
                              **kwargs)

def read_lipo_assay(csv_path=None, **kwargs):
    if csv_path is None:
        csv_path = os.path.join(DATA_PATH, "lipo", "lipo.csv")
    return read_tabular_assay(csv_path,
                              target_col="lipo" ,
                              domain="lipo",
                              **kwargs)


def read_chembl_cell_assay_OOD():
    datas = csv.reader(open(f"{DATA_PATH}/chembl/chembl_processed_chembl32_percent.csv", "r"),
                       delimiter=',')
    assay_id_dicts = {}
    # kd_assay_set = set()
    pic50s = []
    for line in datas:
        unit = line[7]
        assay_id = "{}_{}_{}".format(line[11], line[7], line[8]).replace("/", "_")
        if assay_id not in assay_id_dicts:
            assay_id_dicts[assay_id] = []
        smiles = line[13]
        assay_type = line[10]
        std_type = line[8]
        if std_type != "Activity":
            continue
        unit = line[7]
        std_rel = line[5]
        if std_rel != "=":
            continue
        pic50_exp = math.log10(float(line[6])) - 1.66 + -2.249
        affi_prefix = line[5]
        pic50s.append(pic50_exp)
        ligand_info = {
            "assay_type": std_type,
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": affi_prefix
        }
        assay_id_dicts[assay_id].append(ligand_info)

    print("ood mean", np.mean(pic50s))
    # print(list(kd_assay_set))
    # exit()
    assay_id_dicts_new = {}
    ligands_num = []
    for assay_id, ligands in assay_id_dicts.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.5:
            continue
        if len(ligands) < 20:
            continue
        ligands_num.append(len(ligands))
        assay_id_dicts_new[assay_id] = ligands
    print(np.mean(ligands_num), len(ligands_num))
    assay_ids = list(assay_id_dicts_new.keys())
    return {"ligand_sets": assay_id_dicts_new, "assays": assay_ids}


def read_activity_cliff_assay():
    smiles_as_target = csv.reader(
        open(f"{DATA_PATH}/ACNet/ACNet/ACComponents/ACDataset/data_files/raw_data/all_smiles_target.csv", "r"),
        delimiter=',')

    assay_dicts = {}
    for line in list(smiles_as_target)[1:]:
        smiles = line[0]
        ki = line[1]
        tid = line[2]

        pic50_exp = -math.log10(float(ki))
        ligand_info = {
            "domain": "activity_cliff",
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": ""
        }

        if tid not in assay_dicts:
            assay_dicts[tid] = {}

        assay_dicts[tid][smiles] = ligand_info

    data_few = json.load(
        open(f"{DATA_PATH}/ACNet/ACNet/ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Few.json", "r"))
    data_small = json.load(
        open(f"{DATA_PATH}/ACNet/ACNet/ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Small.json", "r"))
    # data_medium = json.load(
    #     open("../datas/ACNet/ACNet/ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Medium.json", "r"))
    data_all = {**data_few, **data_small}  # , **data_medium}

    assay_dicts_processed = {}
    for tid, data in data_all.items():
        ligands = {}
        for pair in data:
            smiles1 = pair["SMILES1"]
            smiles2 = pair["SMILES2"]
            ligands[smiles1] = assay_dicts[tid][smiles1]
            ligands[smiles2] = assay_dicts[tid][smiles2]
        assay_dicts_processed[tid] = {
            "ligands": ligands,
            "pairs": data
        }

    return assay_dicts_processed


def read_pQSAR_assay():
    filename = f"{DATA_PATH}/pQSAR/ci9b00375_si_002.txt"
    compound_filename = f"{DATA_PATH}/pQSAR/ci9b00375_si_003.txt"
    # first of all, read all the compounds
    compound_file = open(compound_filename, 'r', encoding='UTF-8', errors='ignore')
    clines = compound_file.readlines()
    compound_file.close()

    import numpy as np
    rng = np.random.RandomState(seed=1111)
    compounds = {}
    previous = ''
    previous_id = ''
    for cline in clines:
        cline = str(cline.strip())
        if 'CHEMBL' not in cline:
            if 'Page' in cline or cline == '' or 'Table' in cline or 'SMILE' in cline:
                continue
            else:
                previous += cline
        else:
            strings = cline.split(',')

            if previous_id not in compounds and previous != '':
                compounds[previous_id] = previous.replace('\u2010', '-')

            previous_id = strings[0]
            previous = strings[1]

    compounds[previous_id] = previous.replace('\u2010', '-')

    assay_ids = []
    ligand_set = {}

    file = open(filename, 'r', encoding='UTF-8', errors='ignore')
    lines = file.readlines()
    file.close()

    for line in lines:
        line = str(line.strip())
        if 'CHEMBL' not in line:
            continue
        strings = line.split(' ')
        compound_id = str(strings[0])
        assay_id = int(strings[1])
        try:
            pic50_exp = float(strings[2])
        except:
            pic50_exp = -float(strings[2][1:])
        train_flag = int(strings[4] == "TRN")

        if assay_id not in assay_ids:
            assay_ids.append(assay_id)

        tmp_example = {
            "affi_prefix": "",
            "smiles": compounds[compound_id],
            "pic50_exp": pic50_exp,
            "train_flag": train_flag,
            "domain": "pqsar"
        }

        if assay_id not in ligand_set:
            ligand_set[assay_id] = []
        ligand_set[assay_id].append(tmp_example)

    return {"ligand_sets": ligand_set,
            "assays": list(ligand_set.keys())}


def read_bdb_cross(args):
    BDB_all = read_BDB_per_assay(args)
    save_path = f'{DATA_PATH}/BDB/bdb_split.json'
    split_name_train_val_test = json.load(open(save_path, "r"))
    repeat_ids = set(
        [x.strip() for x in open(f"{DATA_PATH}/BDB/c2b_repeat", "r").readlines()])
    test_ids = [x for x in split_name_train_val_test['test'] if x not in repeat_ids]
    return {"assays": test_ids, "ligand_sets": {aid: BDB_all["ligand_sets"][aid] for aid in test_ids}}


def read_chembl_cross(args):
    chembl_all = read_chembl_assay(args)
    save_path = f'{DATA_PATH}/chembl/chembl_split.json'
    split_name_train_val_test = json.load(open(save_path, "r"))
    repeat_ids = set(
        [x.strip() for x in open(f"{DATA_PATH}/chembl/b2c_repeat", "r").readlines()])
    test_ids = [x for x in split_name_train_val_test['test'] if x not in repeat_ids]
    return {"assays": test_ids, "ligand_sets": {aid: chembl_all["ligand_sets"][aid] for aid in test_ids}}
