import json
import math
import os
import random

import numpy as np
from torch.utils.data import Dataset, sampler, DataLoader
import tqdm
import pickle
import torch
from rdkit import Chem

import copy
from multiprocessing import Pool
from dataset.data_base import BaseMetaDataset


class MetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        self.pre_dir = getattr(args, "preprocessed_dir", None)
        self.use_preprocessed = bool(self.pre_dir and os.path.isdir(self.pre_dir))
        super(MetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        if self.args.datasource == "bdb":
            split_file = os.path.join(self.pre_dir, "bdb_split.json")
        elif self.args.datasource == "chembl":
            split_file = os.path.join(self.pre_dir, "chembl_split.json")
        elif self.args.datasource in ["esol", "lipo", "freesolv", "qm9", "bace",
                                      "bbbp", "hiv", "muv", "clintox", "tox21", "toxcast", "sider"]:
            split_file = os.path.join(self.pre_dir, f"{self.args.datasource}_split.json")
        else:
            print("dataset not exist")
            exit()

        splits = json.load(open(split_file, "r"))

        assay_list = []
        assay_list += splits.get("test", [])
        assay_list += splits.get("valid", [])
        if self.args.train == 1 or self.args.knn_maml:
            assay_list += splits.get("train", [])

        # --- SHARDING: choose only the subset of assay_list assigned to this rank ---
        try:
            ws = int(getattr(self.args, "world_size", os.environ.get("WORLD_SIZE", 1)))
        except Exception:
            ws = 1
        try:
            rk = int(getattr(self.args, "local_rank", os.environ.get("LOCAL_RANK", 0)))
        except Exception:
            rk = 0

        orig_total_assays = len(assay_list)
        if ws is None or ws <= 1:
            shard_assay_list = list(assay_list)
        else:
            shard_assay_list = [assay_list[i] for i in range(rk, orig_total_assays, ws)]

        self.Xs = []
        self.ys = []
        self.smiles_all = []
        self.assaes = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        missing_files = 0
        failed_loads = 0
        cnt = 0
        for aid in shard_assay_list:
            fname = aid.replace("/", "_") + ".pt"
            path = os.path.join(self.pre_dir, fname)
            if not os.path.exists(path):
                missing_files += 1
                continue

            try:
                graphs = torch.load(path, map_location="cpu")
            except Exception:
                failed_loads += 1
                continue

            if not isinstance(graphs, (list, tuple)):
                try:
                    graphs = list(graphs)
                except Exception:
                    failed_loads += 1
                    continue

            # append
            self.Xs.append(graphs)
            try:
                ys = [float(g.y.view(-1)[0]) if hasattr(g, "y") else float(getattr(g, "y", 0.0)) for g in graphs]
            except Exception:
                ys = []
                for g in graphs:
                    try:
                        yv = getattr(g, "y", None)
                        if isinstance(yv, torch.Tensor):
                            ys.append(float(yv.view(-1)[0].item()))
                        else:
                            ys.append(float(yv))
                    except Exception:
                        ys.append(0.0)
            self.ys.append(ys)

            # smiles
            self.smiles_all.append([getattr(g, "smiles", None) for g in graphs])
            self.assaes.append(aid)

            # determine whether this assay is train/val/test in the global splits
            if aid in splits.get("train", []):
                self.train_indices.append(cnt)
            elif aid in splits.get("valid", []):
                self.val_indices.append(cnt)
            else:
                self.test_indices.append(cnt)

            cnt += 1

        # final data_length: reflect counts *loaded by this rank*
        self.data_length = {
            "train": len(self.train_indices),
            "val": len(self.val_indices),
            "test": len(self.test_indices),
            "train_weight": len(self.train_indices),
        }

        print(f"[rank {rk}] >>> Loaded shard {len(self.assaes)} assays (loaded_cnt={cnt})  (orig_total={orig_total_assays})", flush=True)
        print(f"[rank {rk}]     train={len(self.train_indices)}  valid={len(self.val_indices)}  test={len(self.test_indices)}", flush=True)
        if missing_files > 0 or failed_loads > 0:
            print(f"[rank {rk}]  Warning: missing_files={missing_files} failed_loads={failed_loads}", flush=True)
        if len(self.assaes) == 0:
            print(f"[rank {rk}] WARNING: no assays loaded for this shard! Check WORLD_SIZE/LOCAL_RANK and preprocessed_dir.", flush=True)

        return

