import math
import os
import numpy as np
import torch_geometric
from torch.utils.data import Dataset, sampler, DataLoader
from typing import Dict, List, Set, Tuple, Union
import torch
import random
from collections import defaultdict
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def make_mol(s: str, keep_h: bool, add_h: bool, keep_atom_map: bool):
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h if not keep_atom_map else False
    mol = Chem.MolFromSmiles(s, params)

    if add_h:
        mol = Chem.AddHs(mol)

    if keep_atom_map:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        for idx, map_num in enumerate(atom_map_numbers):
            if idx + 1 != map_num:
                new_order = np.argsort(atom_map_numbers).tolist()
                return Chem.rdmolops.RenumberAtoms(mol, new_order)

    return mol


def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    if isinstance(mol, str):
        mol = make_mol(mol, keep_h=False, add_h=False, keep_atom_map=False)
    if isinstance(mol, tuple):
        mol = mol[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    scaffolds = defaultdict(set)
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def preprocess_assay(in_data):
    lines, test_sup_num = in_data
    x_tmp = []
    smiles_list = []
    activity_list = []

    if not lines:
        return None

    if len(lines) > 10000:
        return None

    for line in lines:
        smiles = line["smiles"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Failed to parse SMILES {smiles}")
            continue
        fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fingerprints_vect, fp_numpy)
        pic50_exp = line["pic50_exp"]
        activity_list.append(pic50_exp)
        x_tmp.append(fp_numpy)
        smiles_list.append(smiles)

    x_tmp = np.array(x_tmp).astype(np.float32)
    affis = np.array(activity_list).astype(np.float32)

    domain = lines[0].get("domain", "none")
    if len(x_tmp) < 20 and lines[0].get("domain", "none") in ['chembl', 'bdb', 'pqsar', 'fsmol']:
        return None
    return x_tmp, affis, smiles_list


class BaseMetaDataset(Dataset):
    def __init__(self, args, exp_string):
        self.args = args
        self.current_set_name = "train"
        self.exp_string = exp_string

        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.test_seed,
                          'train_weight': args.train_seed}
        self.batch_size = args.meta_batch_size

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0
        self.split_all = []

        self.current_epoch = 0
        self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def get_split(self, X_in, y_in, is_test=False, sup_num=None, scaffold_split=None, y_opls4=None, rand_seed=None,
                  smiles=None):
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "NA"))
        print(
            f"[{rank}] ENTER get_split: len(X_in)={len(X_in)} is_test={is_test} sim_cut={self.args.similarity_cut} raw_sup={self.args.test_sup_num}",
            flush=True)

        rng_py = random.Random(rand_seed)

        def data_split(data_len, sup_num_, rng_):
            if not is_test:
                min_num = math.log10(max(10, int(0.3 * data_len)))
                max_num = math.log10(int(0.85 * data_len))
                # todo:for few-shot setting
                sup_num_ = random.uniform(min_num, max_num)
                sup_num_ = math.floor(10 ** sup_num_)
            split = [1] * sup_num_ + [0] * (data_len - sup_num_)
            rng_.shuffle(split)
            return np.array(split)

        def data_split_byvalue(y, sup_num_):
            sup_index = np.argpartition(y, sup_num_)[:sup_num_].tolist()
            split = []
            for i in range(len(y)):
                if i in sup_index:
                    split.append(1)
                else:
                    split.append(0)
            return np.array(split)

        def data_split_bysaffold(smiles, sup_num_):
            scaffold_dict = scaffold_to_smiles(smiles, use_indices=True)
            scaffold_id_list = [(k, v) for k, v in scaffold_dict.items()]
            scaffold_id_list = sorted(scaffold_id_list, key=lambda x: len(x[1]))
            idx_list_all = []
            for scaffold, idx_list in scaffold_id_list:
                idx_list_all += idx_list

            sup_index = idx_list_all[:sup_num_]
            split = []
            for i in range(len(y)):
                if i in sup_index:
                    split.append(1)
                else:
                    split.append(0)
            return np.array(split)

        def data_split_bysim(Xs, sup_num_, rng_, sim_cut_):
            def get_sim_matrix(a, b):
                a_bool = (a > 0.).float()
                b_bool = (b > 0.).float()
                and_res = torch.mm(a_bool, b_bool.transpose(0, 1))
                or_res = a.shape[-1] - torch.mm((1. - a_bool), (1. - b_bool).transpose(0, 1))
                sim = and_res / or_res
                return sim

            Xs_torch = torch.tensor(Xs, dtype=torch.float32)
            sim_matrix = get_sim_matrix(Xs_torch, Xs_torch).numpy() - np.eye(len(Xs))

            # safety: ensure sup_num_ is not larger than len(Xs)
            sup_num_ = min(sup_num_, max(1, len(Xs) - 1))

            split = [1] * sup_num_ + [0] * (len(Xs) - sup_num_)
            rng_.shuffle(split)
            sup_index = [i for i, t in enumerate(split) if t == 1]

            split = []
            for i in range(len(Xs)):
                if i in sup_index:
                    split.append(1)  # support
                else:
                    # if sup_index empty (shouldn't happen after above), treat as query
                    if len(sup_index) == 0:
                        split.append(0)
                    else:
                        max_sim = np.max(sim_matrix[i][sup_index])
                        print(f"[{rank}] data_split_bysim: sample={i} max_sim={max_sim:.4f} sim_cut={sim_cut_}",
                              flush=True)
                        if max_sim >= sim_cut_:
                            split.append(-1)  # filter out
                        else:
                            split.append(0)  # query
            return np.array(split)

        # ---------------------- 外层逻辑 ----------------------
        rng = np.random.RandomState(seed=rand_seed)

        if len(X_in) > 64 and not is_test and self.args.datasource != 'gdsc':
            assert y_opls4 is None
            subset_num = 64
            raw_data_len = len(X_in)
            mask = [1] * subset_num + [0] * (raw_data_len - subset_num)
            rng_py.shuffle(mask)
            idxs = [i for i, flag in enumerate(mask) if flag == 1]
            X = [X_in[i] for i in idxs]
            y = [y_in[i] for i in idxs]
        else:
            X, y = X_in, y_in

        # keep copies for fallback if filtering removes everything
        X_pre = list(X) if not isinstance(X, np.ndarray) else list(X.tolist())
        y_pre = list(y)

        # ---------------------- 解析并规范 sup_num ----------------------
        raw_sup = self.args.test_sup_num
        try:
            # allow string like "16" or "0.2"
            if isinstance(raw_sup, str):
                if '.' in raw_sup:
                    raw_sup = float(raw_sup)
                else:
                    raw_sup = int(raw_sup)
            if isinstance(raw_sup, (np.generic,)):
                raw_sup = raw_sup.item()
        except Exception:
            print(f"[DEBUG] could not parse test_sup_num (raw={self.args.test_sup_num}), fallback to 1", flush=True)
            raw_sup = 1

        # interpret proportion or absolute
        if isinstance(raw_sup, float) and raw_sup <= 1.0:
            sup_num = max(1, int(round(raw_sup * len(X))))
        else:
            try:
                sup_num = int(raw_sup)
            except Exception:
                sup_num = 1

        # ensure not larger than len(X)-1 so at least 1 query remains
        sup_num = min(sup_num, max(1, len(X) - 1))

        print(
            f"[DEBUG DATA_SPLIT] raw_test_sup_num={self.args.test_sup_num} parsed={raw_sup} -> sup_num_effective={sup_num} lenX_before={len(X)} sim_cut={self.args.similarity_cut} is_test={is_test}",
            flush=True)

        # ---------------------- 划分并打印 debug ----------------------
        method = None
        if scaffold_split is not None:
            if isinstance(scaffold_split, str):
                method = scaffold_split.lower()
            elif scaffold_split is True:
                method = 'scaffold'
            elif isinstance(scaffold_split, (int, float)):
                method = 'scaffold' if scaffold_split else None

        print(f"[{rank}] DEBUG choose split method={method}", flush=True)

        if method == 'scaffold':
            split = data_split_bysaffold(smiles, sup_num)
        elif method == 'value':
            split = data_split_byvalue(y, sup_num)
        elif getattr(self.args, 'similarity_cut', 1.0) < 1.0:
            assert is_test
            split = data_split_bysim(X, sup_num, rng, self.args.similarity_cut)

            # debug before filter (counts of -1/0/1)
            try:
                binc_pre = np.bincount(split + 1)  # shift: -1->0, 0->1, 1->2
                print(
                    f"[{rank}] DEBUG SPLIT BEFORE FILTER len={len(X)} binc_shifted(-1->0,0->1,1->2)={binc_pre.tolist()}",
                    flush=True)
            except Exception as e:
                print(f"[{rank}] DEBUG split binc_pre error: {e}", flush=True)

            # filter out -1
            X = np.array([t for i, t in enumerate(X) if split[i] != -1])
            y = [t for i, t in enumerate(y) if split[i] != -1]
            split = np.array([t for t in split if t != -1], dtype=np.int32)

            # debug after filter (counts of 0=query,1=support)
            try:
                if len(split) > 0:
                    binc_after = np.bincount(split)
                    print(
                        f"[{rank}] DEBUG SPLIT AFTER FILTER lenX_after={len(X)} binc_after(query=0,support=1)={binc_after.tolist()}",
                        flush=True)
                else:
                    print(f"[{rank}] DEBUG SPLIT AFTER FILTER lenX_after=0 (all dropped by similarity filter)",
                          flush=True)
            except Exception as e:
                print(f"[{rank}] DEBUG split binc_after error: {e}", flush=True)

            # fallback: if all filtered out, restore and enforce minimal split
            if len(split) == 0:
                print(f"[{rank}] DEBUG SPLIT FIX all samples dropped; falling back to original X_pre", flush=True)
                X = np.array(X_pre)
                y = list(y_pre)
                if len(X) == 1:
                    X = np.array([X[0], X[0]])
                    y = [y[0], y[0]]
                split = np.zeros(len(X), dtype=np.int32)
                split[0] = 1
                print(
                    f"[{rank}] DEBUG SPLIT FIX fallback split len={len(split)} sup={int(np.sum(split == 1))} query={int(np.sum(split == 0))}",
                    flush=True)
            else:
                # ensure at least 1 support and 1 query
                n_sup = int(np.sum(split == 1))
                n_query = int(np.sum(split == 0))
                if n_sup == 0 and len(split) > 0:
                    split[0] = 1
                    print(f"[{rank}] DEBUG SPLIT FIX no support after filter; forced split[0]=1", flush=True)
                if n_query == 0 and len(split) > 1:
                    for idx in range(len(split)):
                        if split[idx] == 1:
                            split[idx] = 0
                            if np.sum(split == 1) == 0 and len(split) > 1:
                                split[0] = 1
                            break
                    print(
                        f"[{rank}] DEBUG SPLIT FIX no query after filter; adjusted split to ensure at least one query",
                        flush=True)
        else:
            split = data_split(len(X), sup_num, rng)

        # apply y_opls4 if provided (unchanged)
        if y_opls4 is not None:
            assert len(y_opls4) == len(y)
            y = (1 - split) * y + split * np.array(y_opls4)

        # final debug summary
        try:
            final_sup = int(np.sum(np.array(split) == 1)) if len(split) > 0 else 0
            final_query = int(np.sum(np.array(split) == 0)) if len(split) > 0 else 0
            print(
                f"[{rank}] DEBUG DATA_SPLIT FINAL lenX_final={len(X)} final_sup={final_sup} final_query={final_query}",
                flush=True)
        except Exception as e:
            print(f"[{rank}] DEBUG DATA_SPLIT FINAL count error: {e}", flush=True)

        return [X, y, split]

    def get_set(self, current_set_name, idx):
        # --- 选 si_list 和 ret_weight 如前 ---
        if current_set_name == 'train':
            si_list = self.train_indices[idx * self.batch_size: (idx + 1) * self.batch_size]
            ret_weight = [1.0] * len(si_list)
        elif current_set_name == 'val':
            si_list = [self.val_indices[idx]]
            ret_weight = [1.0]
        elif current_set_name == 'test':
            si_list = [self.test_indices[idx]]
            ret_weight = [1.0]
        elif current_set_name == 'train_weight':
            if self.idxes is not None:
                si_list = self.idxes[idx * self.weighted_batch: (idx + 1) * self.weighted_batch]
                ret_weight = self.train_weight[idx * self.weighted_batch: (idx + 1) * self.weighted_batch]
            else:
                si_list = [self.train_indices[idx]]
                ret_weight = [1.0]
        else:
            raise ValueError(f"Unknown set name: {current_set_name}")

        assay_names = []
        ligands_all = []
        datas = []

        try:
            raw_sup = float(self.args.test_sup_num)
        except:
            raw_sup = 5.0  # 默认值 fallback

        # 逻辑：如果 > 1 则视为绝对数量 (如 5-shot)，如果 < 1 则视为比例 (如 0.2)
        is_ratio = raw_sup < 1.0

        for si in si_list:
            assay_id = self.assaes[si]
            assay_names.append(assay_id)
            ligands_all.append(self.smiles_all[si])

            graphs = self.Xs[si]
            n = len(graphs)
            if n < 2:
                split_mask = torch.zeros(n, dtype=torch.bool)
            else:
                if is_ratio:
                    k = int(n * raw_sup)
                else:
                    k = int(raw_sup)

                # 必须保证至少有1个 support，且至少留1个 query
                k = max(1, min(n - 1, k))

                perm = torch.randperm(n)
                mask = torch.zeros(n, dtype=torch.bool)
                mask[perm[:k]] = True
                split_mask = mask

            y_list = []
            for g in graphs:
                yv = g.y
                y_list.append(yv.view(-1)[0].item() if isinstance(yv, torch.Tensor) else float(yv))
            y_tensor = torch.tensor(y_list, dtype=torch.float32)

            datas.append((graphs, y_tensor, split_mask))

        xs_list, ys_list, splits_list = zip(*datas)
        xs_list = list(xs_list)
        ys_list = list(ys_list)
        splits_list = list(splits_list)

        return xs_list, ys_list, splits_list, assay_names, ret_weight, ligands_all

    def __len__(self):
        if self.current_set_name == "train":
            total_samples = self.data_length[self.current_set_name] // self.args.meta_batch_size
        elif self.current_set_name == "train_weight":
            if self.idxes is not None:
                total_samples = len(self.idxes) // self.weighted_batch
            else:
                total_samples = self.data_length["train_weight"] // self.weighted_batch
        else:
            total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def set_train_weight(self, train_weight=None, idxes=None, weighted_batch=1):
        self.train_weight = train_weight
        self.idxes = idxes
        self.weighted_batch = weighted_batch

    def switch_set(self, set_name, current_epoch=0):
        self.current_set_name = set_name
        self.current_epoch = current_epoch
        if set_name == "train":
            rng = np.random.RandomState(seed=self.init_seed["train"] + current_epoch)
            rng.shuffle(self.train_indices)

    def __getitem__(self, idx):
        return self.get_set(self.current_set_name, idx=idx)


def my_collate_fn(batch):
    batch = batch[0]
    return batch


class SystemDataLoader(object):
    def __init__(self, args, MetaDataset, current_epoch=0, exp_string=None):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_epoch: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.args = args
        self.batch_size = args.meta_batch_size
        self.total_train_epochs = 0
        self.dataset = MetaDataset(args, exp_string=exp_string)
        self.full_data_length = self.dataset.data_length
        self.continue_from_epoch(current_epoch=current_epoch)

    def get_train_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, num_workers=8, shuffle=False, drop_last=True,
                          collate_fn=my_collate_fn)

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=my_collate_fn)

    def continue_from_epoch(self, current_epoch):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_epoch:
        """
        self.total_train_epochs += current_epoch

    def get_train_batches_weighted(self, weights=None, idxes=None, weighted_batch=1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        self.dataset.switch_set(set_name="train_weight", current_epoch=self.total_train_epochs)
        self.dataset.set_train_weight(weights, idxes, weighted_batch=weighted_batch)
        self.total_train_epochs += 1
        return self.get_dataloader()

    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        self.dataset.switch_set(set_name="train", current_epoch=self.total_train_epochs)
        for idx in range(len(self.dataset)):
            data = self.dataset[idx]
            X, y, split, assay, ret_weight, ligands_all = data
            yield (X, y, split, assay, ret_weight,ligands_all)
        self.total_train_epochs += self.batch_size
        return self.get_train_dataloader()

    def get_val_batches(self, total_batches=-1, repeat_cnt=0):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param repeat_cnt:
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="val", current_epoch=repeat_cnt)
        return self.get_dataloader()

    def get_test_batches(self, total_batches=-1, repeat_cnt=0):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param repeat_cnt:
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name='test', current_epoch=repeat_cnt)
        return self.get_dataloader()
