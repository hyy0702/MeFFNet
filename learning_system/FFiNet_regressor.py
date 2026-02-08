import torch
import gc
import sys
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen

from .system_base import RegressorBase
from torch_geometric.data import Data
from .FFiNet_model import FFiNetModel
from .model_utils import MLP
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch.nn.utils.stateless import functional_call


def get_atom_features(atom):
    features = []
    # 原子类型（12维）
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Other']
    features += [1 if atom.GetSymbol() == t else 0 for t in atom_types]

    # 杂化类型（5维）
    hybridization = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    features += [int(atom.GetHybridization() == h) for h in hybridization]

    # 其他特征（芳香性、环内、形式电荷等）
    features += [
        atom.GetIsAromatic(),
        atom.IsInRing(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetDegree()
    ]

    # Gasteiger Partial Charge (电荷)
    try:
        charge = float(atom.GetFormalCharge())
        if math.isnan(charge) or math.isinf(charge):
            charge = 0.0
    except:
        charge = 0.0
    features.append(charge)

    # Crippen LogP Contribution (LogP 贡献)
    contrib = 0.0
    try:
        val = Crippen.ContribTable[Crippen.GetAtomCorr(atom)]
        if not math.isnan(val) and not math.isinf(val):
            contrib = float(val)
    except:
        contrib = 0.0
    features.append(contrib)

    # 填充至66维
    while len(features) < 66:
        features.append(0)

    return features[:66]

class FFiNetRegressor(RegressorBase):
    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)
        self.use_preprocessed = getattr(args, 'preprocessed_dir', None) is not None
        self.regressor = FFiNetModel(args, input_shape)
        self.atom_feat_dim = 66  # 原子特征维度

        self.feature_per_layer = args.feature_per_layer
        self.pred_hidden_dim = args.pred_hidden_dim
        self.pred_layers = args.pred_layers
        self.num_tasks = args.num_tasks
        self.pred_dropout = args.pred_dropout

        self.edge_index_start = 66  # 边索引起始位置
        self.edge_index_end = 68   # 边索引结束位置（不包含）
        self.pos_start = 68        # 3D坐标起始位置
        self.pos_end = 71          # 3D坐标结束位置（不包含）
        self.predict = MLP(
            [self.feature_per_layer[-1] * 4] +
            [self.pred_hidden_dim] * self.pred_layers +
            [self.num_tasks],  # 确保输出维度为 [batch_size, num_tasks]
            dropout=self.pred_dropout
        )

        self.post_init(args)
        # 动态划分参数
        self.random_seed = 42  # 显式定义随机种子

    def _model_device(self):
        model = self.regressor
        # if wrapped in DDP/DataParallel, the actual module is in .module
        if hasattr(model, 'module'):
            model = model.module
        try:
            p = next(model.parameters())
            return p.device
        except StopIteration:
            return torch.device('cpu')

    def _dynamic_split(self, num_samples):
        # 生成随机索引
        indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(self.random_seed))
        train_size = int(num_samples * self.train_ratio)

        # 创建 splits 张量（1 表示训练，0 表示测试）
        splits = torch.zeros(num_samples, dtype=torch.float32)
        splits[indices[:train_size]] = 1.0
        return splits

    def generate_split(self, x_task):
        num_samples = len(x_task)
        support_num = int(self.args.test_sup_num)
        support_num = min(support_num, max(1, num_samples - 1))

        split = torch.zeros(num_samples, dtype=torch.bool)
        split[:support_num] = True
        split = split[torch.randperm(num_samples, generator=torch.Generator().manual_seed(42))]
        return split

    def forward(self, data_batch, epoch, num_steps, is_training_phase, **kwargs):
        xs, ys, splits, assay_idxes, assay_weights, _ = data_batch
        total_losses, per_task_target_preds, final_weights, per_task_metrics = [], [], [], []

        # 对 meta‑batch 中的每个任务并行处理
        for x_task, y_task, split, assay_idx, w in zip(xs, ys, splits, assay_idxes, assay_weights):
            device = self._model_device()
            if isinstance(split, torch.Tensor) and split.device != device:
                split = split.to(device)

            # inner‑loop：支持集多步 + 返回 query preds
            names_w, support_losses, task_losses, query_preds = self.inner_loop(
                x_task, y_task, assay_idx, split,
                is_training_phase, epoch, num_steps
            )
            # task_losses 最后一项即 query_loss
            query_loss = task_losses[-1]

            # 收集预测、metrics、loss、最终权重
            per_task_target_preds.append(query_preds.detach().cpu().numpy())
            metrics = self.get_metric(y_task, query_preds, split)
            # 调试：把本任务的指标都打印出来
            # print(f"[METRIC] assay={assay_idx}  rmse={metrics['rmse']:.4f}  "
            #       f"r2={metrics['r2']:.4f}  R2os={metrics['R2os']:.4f}  "
            #       f"y_train_mean={metrics['y_train_mean']:.4f}")
            metrics["each_step_loss"] = support_losses
            per_task_metrics.append(metrics)

            total_losses.append(query_loss)
            final_weights.append(names_w)

        # 在所有任务上做一次 meta‑aggregate
        losses = self.get_across_task_loss_metrics(total_losses, assay_weights)
        if isinstance(losses, dict) and isinstance(losses.get('loss'), torch.Tensor):
            l = losses['loss']

        return losses, per_task_target_preds, final_weights, per_task_metrics

    def net_forward(self, x, y, assay_idx, split, weights, is_support, backup_running_statistics, training, num_step):
        device = self._model_device()

        if isinstance(x, (list, tuple)) and len(x) == 0:
            return torch.tensor(0.0, device=device), torch.zeros(0, device=device)

        # 1) 准备图批次
        if isinstance(x, list):
            batch = Batch.from_data_list(x).to(device)
        elif isinstance(x, Data):
            batch = x.to(device)
        else:
            try:
                if isinstance(x, np.ndarray):
                    x = x.tolist()
                    if len(x) == 0:
                        return torch.tensor(0.0, device=device), torch.zeros(0, device=device)
                    batch = Batch.from_data_list(list(x)).to(device)
                else:
                    raise ValueError
            except Exception:
                raise ValueError(f"Unsupported x type {type(x)} in net_forward (assay_idx={assay_idx})")

        # 2) 处理标签和 split
        y = y.to(device).view(-1, 1)
        split = split.to(device).flatten()

        mask = split.bool() if is_support else (~split).bool()

        # 3) 调用模型前向并注入快照参数
        regressor_module = getattr(self.regressor, "module", self.regressor)

        if weights is None:
            preds = regressor_module.forward(
                batch,
                weights=None,
                training=training,
                backup_running_statistics=backup_running_statistics,
                num_step=num_step
            )
        else:
            # 对齐 weights key 到 model.state_dict() 的 keys
            model_keys = list(regressor_module.state_dict().keys())

            def align_weight_keys(weights_dict, model_keys):
                if set(weights_dict.keys()) == set(model_keys):
                    return weights_dict
                mapped = {}
                for k, v in weights_dict.items():
                    k_strip = k[len("module."):] if k.startswith("module.") else k
                    if k in model_keys:
                        mapped[k] = v
                    elif k_strip in model_keys:
                        mapped[k_strip] = v
                    else:
                        tail = k.split(".", 1)[-1]
                        for mk in model_keys:
                            if mk.endswith(tail):
                                mapped[mk] = v
                                break
                        else:
                            mapped[k] = v
                return mapped

            mapped_weights = align_weight_keys(weights, model_keys)

            # functional_call 的正确调用： third arg 是 args tuple，fourth arg 是 kwargs dict
            preds = functional_call(
                regressor_module,
                mapped_weights,
                (batch,),
                {
                    "training": training,
                    "backup_running_statistics": backup_running_statistics,
                    "num_step": num_step
                }
            )
        # -------------------------------------------------

        if mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device), preds
        assert mask.numel() == preds.shape[0], (
            f"mask length {mask.numel()} != num_graphs {preds.shape[0]}"
        )

        # ensure mask is boolean tensor on same device as preds
        mask_bool = mask.to(preds.device).bool()

        # sanity checks for shapes
        if mask_bool.numel() != preds.shape[0]:
            # print debug and try to adapt
            print(
                f"[DEBUG net_forward] mask length {mask_bool.numel()} != num_graphs {preds.shape[0]} (assay_idx={assay_idx})",
                flush=True)
            # try to coerce if possible
            if mask_bool.numel() == 1:
                mask_bool = mask_bool.repeat(preds.shape[0])
            else:
                # fallback: use all True
                mask_bool = torch.ones(preds.shape[0], dtype=torch.bool, device=preds.device)

        # call _compute_loss with long mask (existing API expects long for split arg)
        loss, _ = self._compute_loss(preds, y.to(preds.device), mask_bool.long())

        del batch, mask_bool

        return loss, preds


    def run_validation_iter(self, raw_batch):
        self.is_training_phase = False
        if self.training:
            self.eval()

        # move to model device
        device = self._model_device() if hasattr(self, "_model_device") else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        task_type = getattr(self.args, "task_type", "regression")

        zero_loss = torch.tensor(0.0, device=device)
        dummy_metrics = {"r2": 0.0, "rmse": 0.0, "R2os": 0.0,
                         "each_step_loss": [], "y_train_mean": 0.0, "pred": [], "true": []}
        def get_dummy():
            return {'loss': zero_loss}, [], [], [dummy_metrics]

        try:
            data_batch = self._preprocess_batch(raw_batch)
        except Exception:
            return get_dummy()

        if isinstance(data_batch, dict) and data_batch.get('empty', False):
            return get_dummy()

        # unpack
        xs, ys, splits, assay_idxes, assay_weight, smiless = data_batch
        if not isinstance(xs, (list, tuple)) or len(xs) == 0:
            return get_dummy()

        num_mols = len(xs)
        if num_mols == 0:
            return get_dummy()

        # Build Batch
        try:
            batch_all = Batch.from_data_list(xs).to(device)
        except Exception:
            return get_dummy()

        # y_all
        y_all = torch.stack([torch.as_tensor(y) if not torch.is_tensor(y) else y for y in ys]).squeeze().to(device)
        if y_all.dim() > 1:
            y_all = y_all.view(-1)

        # Build split mask
        if torch.is_tensor(splits[0]) and splits[0].dim() == 1 and splits[0].numel() == num_mols:
            split_all = splits[0].to(device).bool()
        else:
            # fallback: first element support
            split_all = torch.zeros(num_mols, dtype=torch.bool, device=device)
            split_all[0] = True

        query_mask = ~split_all

        n_updates = int(getattr(self.args, "num_updates", 1))
        with torch.no_grad():
            y_noadapt = self.regressor.forward(batch_all,
                                               weights=None,
                                               training=False,
                                               backup_running_statistics=False,
                                               num_step=0)
            y_adapt = self.regressor.forward(batch_all,
                                             weights=None,
                                             training=False,
                                             backup_running_statistics=False,
                                             num_step=n_updates)

        # normalize shapes to 1D
        def to_1d(t):
            if torch.is_tensor(t):
                if t.dim() > 1 and t.size(1) == 1:
                    return t.view(-1)
                return t.view(-1)
            return torch.tensor(t, dtype=torch.float32, device=device).view(-1)

        y0 = to_1d(y_noadapt)
        y1 = to_1d(y_adapt)

        # choose adapted predictions as final
        if y0.numel() != y1.numel():
            y_pred_all = y1
        else:
            y_pred_all = y1

        # flatten if necessary
        if y_pred_all.dim() > 1 and y_pred_all.size(1) == 1:
            y_pred_all = y_pred_all.view(-1)

        # support-based scale + bias calibration
        eps = 1e-8
        if task_type == "regression" and split_all.sum().item() > 0:
            y_sup = y_all[split_all]
            p_sup = y_pred_all[split_all]
            bias = (y_sup - p_sup).mean()
            y_pred_all_corrected = y_pred_all + bias

            y_sup_mean = y_sup.mean()
            y_sup_std = y_sup.std(unbiased=False) if y_sup.numel() > 1 else torch.tensor(0.0, device=device)

            p_sup_mean = p_sup.mean()
            p_sup_std = p_sup.std(unbiased=False) if p_sup.numel() > 1 else torch.tensor(0.0, device=device)

        else: # classification
            y_pred_all_corrected = y_pred_all

        # compute metrics on query positions
        y_true_tensor = y_all[query_mask]
        y_pred_tensor = y_pred_all_corrected[query_mask]

        metrics = self.get_metric(y_all, y_pred_all_corrected, split_all)

        # === 将 Query Set 的 SMILES 注入 metrics ===
        try:
            # 只有当 smiless 存在且不为空时才处理
            if smiless is not None and len(smiless) > 0:
                # split_all: True为Support, False为Query
                if torch.is_tensor(split_all):
                    mask_np = split_all.detach().cpu().numpy().astype(bool)
                else:
                    mask_np = np.array(split_all).astype(bool)

                # 筛选出 Query Set 的 SMILES (即 mask 为 False 的部分)
                # smiless 通常是一个 list，我们需要逐个筛选
                # 确保长度一致
                if len(smiless) == len(mask_np):
                    query_smiless = [s for s, is_support in zip(smiless, mask_np) if not is_support]
                    metrics['smiles'] = query_smiless
                else:
                    # 如果长度对不上（罕见），为了不报错，填充空字符串
                    # 计算 query 的数量
                    query_count = (~mask_np).sum()
                    metrics['smiles'] = ["Error_Len_Mismatch"] * query_count
        except Exception as e:
            print(f"[Warning] Failed to extract SMILES: {e}")
            metrics['smiles'] = []
        # ===========================================

        return {'loss': zero_loss}, None, None, [metrics]

    def run_train_iter(self, raw_batch, epoch):
        self.is_training_phase = True
        epoch = int(epoch)

        if not self.training:
            self.train()

        device = self._model_device() if hasattr(self, "_model_device") else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        zero_loss = torch.tensor(0.0, device=device)

        # debug: control how many times we print verbose diagnostics
        debug_limit = getattr(self, "_debug_print_limit", 5)
        debug_count = getattr(self, "_debug_print_count", 0)
        do_debug_print = debug_count < debug_limit

        data_batch = raw_batch
        if isinstance(data_batch, dict) and data_batch.get('empty', False):
            # 这个任务没有任何分子，跳过
            print("[SKIP] inner-loop: no valid molecules in this task")
            return {'loss': None}, []

        # 1) 运行 forward 并拿到 final_weights 列表
        losses, per_task_target_preds, final_weights, _ = self.forward(
            data_batch=data_batch,
            epoch=epoch,
            num_steps=self.args.num_updates,
            is_training_phase=True
        )

        loss = None
        try:
            loss = losses.get('loss', None) if isinstance(losses, dict) else None
        except Exception:
            loss = None
        kname = "layer_dict.linear.weights".replace(".", "-")

        # 4) Meta‑update：只有当 loss 可微时才更新
        do_meta_update = isinstance(loss, torch.Tensor) and loss.requires_grad

        if do_meta_update:
            total_loss = loss.sum()
            # 可选的 lrloss 正则（仅在 epoch 达到阈值时）
            if epoch >= getattr(self.args, "begin_lrloss_epoch", 0):
                try:
                    lr_param = self.inner_loop_optimizer.names_learning_rates_dict.get(kname, None)
                    if lr_param is not None:
                        total_loss = total_loss + torch.sum(lr_param * lr_param) * getattr(self.args, "lrloss_weight", 0.0)
                except Exception:
                    # 忽略 lr 正则相关错误，继续 meta update
                    pass
            # 执行 meta update（外部代码会处理 optimizer/scheduler）
            try:
                self.meta_update(loss=total_loss)
            except Exception as e:
                print(f"[WARN] meta_update failed: {e}", flush=True)
            try:
                self.scheduler.step()
            except Exception:
                pass
        else:
            # 输出原因，便于调试
            if loss is None:
                print("[SKIP] meta_update: loss is None")
            elif isinstance(loss, (float, int)):
                print("[SKIP] meta_update: loss is numeric (float/int), not a differentiable tensor")
            elif isinstance(loss, torch.Tensor) and not loss.requires_grad:
                print("[SKIP] meta_update: loss is a tensor but requires_grad is False")
            else:
                print("[SKIP] meta_update: unknown loss type, skipping", type(loss))

        return losses, per_task_target_preds

    def _preprocess_batch(self, data_batch):
        xs_fp, ys_fp, splits_fp, assay_idxes, assay_weights, smiless = data_batch

        # --- unwrap common single-element outer wrappers (DataLoader with batch_size=1) ---
        try:
            if isinstance(xs_fp, (list, tuple)) and len(xs_fp) == 1:
                xs_fp = xs_fp[0]
        except Exception:
            pass
        try:
            if isinstance(ys_fp, (list, tuple)) and len(ys_fp) == 1:
                ys_fp = ys_fp[0]
        except Exception:
            pass
        try:
            if isinstance(splits_fp, (list, tuple)) and len(splits_fp) == 1:
                splits_fp = splits_fp[0]
        except Exception:
            pass
        try:
            if isinstance(assay_idxes, (list, tuple)) and len(assay_idxes) == 1:
                assay_idxes = assay_idxes[0]
        except Exception:
            pass
        try:
            if isinstance(assay_weights, (list, tuple)) and len(assay_weights) == 1:
                assay_weights = assay_weights[0]
        except Exception:
            pass
        try:
            if isinstance(smiless, (list, tuple)) and len(smiless) == 1 and isinstance(smiless[0], (list, tuple)):
                smiless = smiless[0]
        except Exception:
            pass

        # xs
        if isinstance(xs_fp, (list, tuple)) and len(xs_fp) > 0 and isinstance(xs_fp[0], (list, tuple)):
            xs_list = list(xs_fp[0])
        else:
            if not isinstance(xs_fp, (list, tuple)):
                return {'empty': True}
            xs_list = list(xs_fp)
        n = len(xs_list)
        if n < 2:
            return {'empty': True}

        # determine device
        if hasattr(self, "_model_device"):
            device = self._model_device()
        else:
            device = getattr(self, "device", None)
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- build ys_list ---
        ys_list = []
        try:
            if torch.is_tensor(ys_fp):
                if ys_fp.dim() == 1 and ys_fp.numel() == n:
                    for v in ys_fp:
                        ys_list.append(v.to(device).float().view(-1))
                else:
                    ys_list.append(ys_fp.to(device).float().view(-1))
            elif isinstance(ys_fp, (list, tuple)):
                for y in ys_fp:
                    if torch.is_tensor(y):
                        ys_list.append(y.to(device).float().view(-1))
                    else:
                        ys_list.append(torch.tensor(y, dtype=torch.float32, device=device).view(-1))
            else:
                ys_list.append(torch.tensor(ys_fp, dtype=torch.float32, device=device).view(-1))
        except Exception:
            return {'empty': True}

        # --- build splits_list ---
        splits_list = []
        try:
            if torch.is_tensor(splits_fp):
                if splits_fp.dim() == 1 and splits_fp.numel() == n:
                    splits_list = [splits_fp.to(device).bool().view(-1)]
                else:
                    m = torch.zeros(n, dtype=torch.bool, device=device)
                    m[0] = True
                    splits_list = [m]
            elif isinstance(splits_fp, (list, tuple)):
                for s in splits_fp:
                    if torch.is_tensor(s):
                        splits_list.append(s.to(device).bool().view(-1))
                    else:
                        splits_list.append(torch.tensor(s, dtype=torch.bool, device=device).view(-1))
                if len(splits_list) == 0:
                    m = torch.zeros(n, dtype=torch.bool, device=device);
                    m[0] = True
                    splits_list = [m]
            else:
                m = torch.zeros(n, dtype=torch.bool, device=device);
                m[0] = True
                splits_list = [m]
        except Exception:
            m = torch.zeros(n, dtype=torch.bool, device=device);
            m[0] = True
            splits_list = [m]

        # final simple sanity
        try:
            if not torch.is_tensor(splits_list[0]) or splits_list[0].numel() != n:
                m = torch.zeros(n, dtype=torch.bool, device=device);
                m[0] = True
                splits_list = [m]
        except Exception:
            m = torch.zeros(n, dtype=torch.bool, device=device);
            m[0] = True
            splits_list = [m]

        # -------------------------------------------------------------------------
        try:
            # consider primary split mask (splits_list[0])
            primary_mask = splits_list[0]
            support_count = int(primary_mask.sum().item())
            query_count = int((~primary_mask).sum().item())
            if support_count == 0 or query_count == 0:
                # upstream will detect this and skip meta update
                return {'empty': True}
        except Exception:
            # if something goes wrong counting (shouldn't), fall back to returning data
            pass

        return xs_list, ys_list, splits_list, assay_idxes, assay_weights, smiless

    def _mol_to_graph_data(self, mol_h: Chem.Mol, y_val: float) -> Data:
        # 节点特征
        feats = [get_atom_features(a) for a in mol_h.GetAtoms()]
        x = torch.tensor(feats, dtype=torch.float)

        # 边索引
        edges = []
        for b in mol_h.GetBonds():
            i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges = edges + [[i,j],[j,i]]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2,0), dtype=torch.long)

        # 角度三元组
        triples = []
        for b in mol_h.GetBonds():
            a1,a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            for nb in mol_h.GetAtomWithIdx(a1).GetNeighbors():
                c = nb.GetIdx()
                if c!=a2: triples.append([c,a1,a2])
            for nb in mol_h.GetAtomWithIdx(a2).GetNeighbors():
                c = nb.GetIdx()
                if c!=a1: triples.append([a1,a2,c])
        if triples:
            triple_index = torch.tensor(triples, dtype=torch.long).t().contiguous()
        else:
            triple_index = torch.zeros((3,0), dtype=torch.long)

        # 四元组（二面角）
        quads = []
        for b in mol_h.GetBonds():
            a1,a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            for na in mol_h.GetAtomWithIdx(a1).GetNeighbors():
                c = na.GetIdx()
                if c==a2: continue
                for nb in mol_h.GetAtomWithIdx(a2).GetNeighbors():
                    d = nb.GetIdx()
                    if d==a1: continue
                    quads.append([c,a1,a2,d])
        if quads:
            quadra_index = torch.tensor(quads, dtype=torch.long).t().contiguous()
        else:
            quadra_index = torch.zeros((4,0), dtype=torch.long)

        # 3D 坐标
        pos = torch.tensor(mol_h.GetConformer().GetPositions(), dtype=torch.float)

        # 边属性都设为 bonded=True
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.bool)

        # 分子级 label 下推到每个原子
        y_atom = torch.full((x.size(0),), y_val, dtype=torch.float)

        # batch
        batch = torch.zeros(x.size(0), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            triple_index=triple_index,
            quadra_index=quadra_index,
            pos=pos,
            edge_attr=edge_attr,
            y=y_atom,
            batch=batch
        )
